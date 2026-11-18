"""CUDA reference and first pipeline stage for the single-file Qwen3 checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from transformers.integrations.accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

DTYPE = torch.float16
DEVICE = torch.device("cuda:0")
PROMPT = "The capital of France is"
DECODE_STEPS = 8


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def check_cache(cache, count, length, dtype):
    assert len(cache.layers) == count
    for layer in cache.layers:
        assert layer.get_seq_length() == length
        for tensor in (layer.keys, layer.values):
            assert tensor.device == DEVICE and tensor.dtype == dtype


class Prefix:
    def __init__(self, model_path):
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.split = config.num_hidden_layers // 2
        config.num_hidden_layers = self.split
        config.layer_types = config.layer_types[: self.split]
        config._attn_implementation = "eager"

        # Keep the HF forward (RoPE, masking and cache), ending at the last
        # decoder layer's residual + MLP sum. Final RMSNorm belongs to the NPU.
        with init_empty_weights(include_buffers=False):
            self.model = Qwen3Model(config)
            self.model.norm = torch.nn.Identity()
        with safe_open(Path(model_path) / "model.safetensors", framework="pt") as weights:
            state = {
                name: weights.get_tensor("model." + name).to(device=DEVICE, dtype=DTYPE)
                for name in self.model.state_dict()
            }
        self.model.load_state_dict(state, strict=True, assign=True)
        # Rotary embedding has a nonpersistent float32 buffer.
        self.model.to(device=DEVICE).eval()
        assert all(p.device == DEVICE and p.dtype == DTYPE for p in self.model.parameters())
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        self.embedding_parameters = self.model.embed_tokens.weight.numel()
        emit("gpu_prefix", device=torch.cuda.get_device_name(DEVICE), cuda_index=0,
             dtype=str(DTYPE), layers=list(range(self.split)), parameters=self.num_parameters,
             embed_tokens=self.embedding_parameters, norm="NPU", lm_head="NPU")
        self.reset()

    def reset(self):
        self.cache = DynamicCache(config=self.model.config)
        self.pos = 0

    @torch.inference_mode()
    def step(self, token_ids: list[int], start: int) -> torch.Tensor:
        if start != self.pos or not token_ids:
            raise ValueError(f"expected input starting at {self.pos}, got {start} and {token_ids}")
        input_ids = torch.tensor([token_ids], device=DEVICE)
        end = start + len(token_ids)
        pos = torch.arange(start, end, device=DEVICE)
        result = self.model(input_ids=input_ids, cache_position=pos, position_ids=pos[None],
                            past_key_values=self.cache, use_cache=True)
        assert result.past_key_values is self.cache
        hidden = result.last_hidden_state
        assert hidden.device == DEVICE and hidden.dtype == DTYPE
        check_cache(self.cache, self.split, end, DTYPE)
        self.pos = end
        return hidden.contiguous().cpu()


@torch.inference_mode()
def reference(model_path, out):
    model_path = Path(model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=DTYPE,
                                                 attn_implementation="eager",
                                                 local_files_only=True).to(DEVICE).eval()
    assert all(p.device == DEVICE and p.dtype == DTYPE for p in model.parameters())
    tied = (model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr())
    assert tied == model.config.tie_word_embeddings
    # The checkpoint stores the tied tensor under both names. The NPU suffix
    # may use either name; verify equality before comparing the two paths.
    if tied:
        with safe_open(model_path / "model.safetensors", framework="pt") as weights:
            if "lm_head.weight" in weights.keys():
                assert torch.equal(weights.get_tensor("lm_head.weight"),
                                   weights.get_tensor("model.embed_tokens.weight"))
    emit("reference", model=str(model_path), model_class=type(model).__name__,
         device=torch.cuda.get_device_name(DEVICE), cuda_index=0, dtype=str(DTYPE),
         torch=torch.__version__, cuda=torch.version.cuda, layers=model.config.num_hidden_layers,
         parameters=sum(p.numel() for p in model.parameters()), embedding_head_tied=tied,
         duplicated_across_stages=model.get_input_embeddings().weight.numel() if tied else 0,
         prompt=PROMPT, prompt_ids=prompt_ids)
    input_ids = prompt_ids
    logits, tokens = [], []
    start = 0
    cache = DynamicCache(config=model.config)
    for step in range(DECODE_STEPS + 1):
        end = start + len(input_ids)
        pos = torch.arange(start, end, device=DEVICE)
        result = model(input_ids=torch.tensor([input_ids], device=DEVICE), past_key_values=cache,
                       position_ids=pos[None], cache_position=pos, use_cache=True, logits_to_keep=1)
        check_cache(cache, model.config.num_hidden_layers, end, DTYPE)
        row = result.logits[0, -1]
        assert row.dtype == DTYPE and row.device == DEVICE and row.isfinite().all()
        logits.append(row.cpu())
        tokens.append(row.argmax().item())
        emit("reference_step", step=step, positions=[start, end - 1], input_ids=input_ids,
             next_token=tokens[-1], kv_layers=len(cache.layers), kv_length=end,
             kv_shape=list(cache.layers[0].keys.shape))
        start = end
        input_ids = [tokens[-1]]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_path=str(model_path), prompt=PROMPT, prompt_ids=prompt_ids,
                    tokens=tokens, logits=torch.stack(logits), dtype="float16",
                    history="Greedy top-1 from the preceding reference step"), out)
    emit("reference_saved", path=str(out), tokens=tokens, text=tokenizer.decode(tokens))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3-0.6B snapshot directory")
    parser.add_argument("--out", default="logs/reference.pt")
    args = parser.parse_args()
    reference(args.model, args.out)

"""Compile Qwen3's upper decoder blocks and run both phases on one RBLN device."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import rebel
import torch
from optimum.rbln import RBLNQwen3ForCausalLM, RBLNQwen3ForCausalLMConfig
from safetensors import safe_open
from transformers import AutoConfig, Qwen3ForCausalLM

# Import the compile-context factory from the pinned main checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vllm_rbln.compilation import create_compile_context

BUNDLE = Path(__file__).resolve().parent / "build" / "suffix"
CHUNK = 64  # The smallest prefill bucket supported by the Qwen3 RBLN wrapper.


def load_suffix(model_path):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    total = config.num_hidden_layers
    start = total // 2
    config.num_hidden_layers = total - start
    config.layer_types = config.layer_types[start:]
    # Local attention indices 0..13 address only this stage's KV, as in patches/attention.py.
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    model.model.embed_tokens = torch.nn.Identity()
    weights = {}
    with safe_open(Path(model_path) / "model.safetensors", framework="pt") as ckpt:
        for name in model.state_dict():
            source = name
            if name.startswith("model.layers."):
                _, _, ix, tail = name.split(".", 3)
                source = f"model.layers.{int(ix) + start}.{tail}"
            elif name == "lm_head.weight" and config.tie_word_embeddings:
                source = "model.embed_tokens.weight"
            weights[name] = ckpt.get_tensor(source).to(torch.float16)
    model.load_state_dict(weights, assign=True, strict=True)
    counts = {label: sum(p.numel() for p in module.parameters()) for label, module in
              [("blocks", model.model.layers), ("norm", model.model.norm), ("lm_head", model.lm_head)]}
    print("suffix weights", {"layers": list(range(start, total)), **counts,
                             "total": sum(counts.values()), "dtype": str(model.dtype)}, flush=True)
    return model.eval(), start, total, counts


@torch.inference_mode()
def compile_suffix(model_path, device=0, bundle=BUNDLE):
    model, start, total, counts = load_suffix(model_path)
    target = rebel.get_npu_name(device)
    sdk_config = RBLNQwen3ForCausalLMConfig(
        use_inputs_embeds=True, use_attention_mask=False, max_seq_len=CHUNK,
        prefill_chunk_size=CHUNK, batch_size=1, num_devices=1, npu=target, device=device,
    )
    cls = RBLNQwen3ForCausalLM
    config = cls.update_rbln_config([], model, model.config, sdk_config)
    wrapped = cls._wrap_model_if_needed(model, config)
    # The same native static KV addresses and weights are shared by both executors.
    context = create_compile_context(use_weight_sharing=True)
    caches = {}
    bundle = Path(bundle)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        raise FileExistsError(f"Choose a fresh build directory: {bundle}")
    with tempfile.TemporaryDirectory(dir=bundle.parent) as tmp:
        out = Path(tmp) / "suffix"
        out.mkdir()
        for phase, cc in zip(("prefill", "decode"), config.compile_cfgs, strict=True):
            names = [name for name, _, _ in cc.input_info if name.startswith("past_key_values_")]
            if phase == "prefill":
                args = cc.get_dummy_inputs(fill=0, meta_tensor_names=names)
                caches = {name: x for (name, _, _), x in zip(cc.input_info, args) if name in names}
                for name, x in caches.items():
                    context.mark_static_address(x, name)
            else:
                args = cc.get_dummy_inputs(fill=0, static_tensors=caches)
            assert len(caches) == 2 * (total - start)
            print("native compile", phase, target, "inputs", cc.input_info[:len(cc.input_info)-len(caches)],
                  "resident K/V", len(caches), cc.input_info[-1], flush=True)
            cm = cls._compile_model(wrapped, cc, args, context, config, phase=phase)
            dest = out / f"{phase}.rbln"
            cm.save(dest)
            print("compiled", phase, dest.name, dest.stat().st_size, "bytes", flush=True)
        metadata = dict(model_path=str(Path(model_path).resolve()), npu=target, layers=[start, total],
                        counts=counts, hidden_size=model.config.hidden_size, vocab_size=model.config.vocab_size)
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        out.rename(bundle)


class Suffix:
    def __init__(self, model_path, device=0, bundle=BUNDLE):
        bundle = Path(bundle)
        meta = json.loads((bundle / "metadata.json").read_text())
        if meta["model_path"] != str(Path(model_path).resolve()) or meta["npu"] != rebel.get_npu_name(device):
            raise ValueError("Model path or NPU target differs from the compile metadata")
        self.hidden = meta["hidden_size"]
        self.vocab = meta["vocab_size"]
        self.position = 0
        self.models = [rebel.RBLNCompiledModel(bundle / f"{phase}.rbln") for phase in ("prefill", "decode")]
        self.prefill, self.decode = [cm.create_runtime(device=device, tensor_type="pt") for cm in self.models]
        print("native runtimes", os.getpid(), "rbln", device, meta, type(self.prefill).__name__,
              file=sys.stderr, flush=True)
        for phase, cm in zip(("prefill", "decode"), self.models):
            device_cache = {key: size for key, size in cm.exp_get_dram_tensor_sizes().items()
                            if "past_key_values" in key}
            if phase == "prefill":
                start, end = meta["layers"]
                cache_names = {f"past_key_values_{i}" for i in range(2 * (end - start))}
                if device_cache.keys() != cache_names:
                    raise RuntimeError("Incomplete native KV allocation for the suffix layers")
            stats = cm.compiled_model_stats.stats
            if len(stats) != 1 or stats[0].device_func_count != 1 or stats[0].host_op_reason:
                raise RuntimeError(f"Expected one native graph without host graph breaks: {stats}")
            print("native graph", phase, "device_alloc_bytes", cm.get_alloc_per_node(),
                  "newly_allocated_KV_buffers", len(device_cache), "stats", stats[0],
                  file=sys.stderr, flush=True)

    @torch.inference_mode()
    def __call__(self, hidden, start_pos):
        length = hidden.shape[1]
        if (hidden.device.type != "cpu" or hidden.dtype != torch.float16
                or hidden.shape != (1, length, self.hidden) or length < 1 or start_pos + length > CHUNK):
            raise ValueError("Invalid CPU hidden states or sequence length")
        runtime = self.prefill if start_pos == 0 else self.decode
        runtime.flush_reports()
        if start_pos == 0:
            padded = torch.nn.functional.pad(hidden, (0, 0, 0, CHUNK - length))
            logits = self.prefill(padded.contiguous(), torch.arange(CHUNK, dtype=torch.int32)[None],
                                  torch.zeros(1, dtype=torch.int16), torch.tensor(length-1, dtype=torch.int16))
        else:
            if start_pos != self.position or length != 1:
                raise ValueError("Decode position must follow the preceding native execution")
            logits = self.decode(hidden.contiguous(), torch.tensor([[start_pos]], dtype=torch.int32),
                                 torch.zeros((1, 1), dtype=torch.int16))
        self.position = start_pos + length
        device_us = runtime.get_elapsed_device_times()
        if os.environ.get("RBLN_RUNTIME_TIMER") == "1" and device_us <= 0:
            raise RuntimeError("Native runtime returned no positive device execution timer")
        print("native runtime returned", "position", start_pos, "input", list(hidden.shape),
              "logits", list(logits.shape), str(logits.dtype), "device_us", device_us,
              file=sys.stderr, flush=True)
        return logits.reshape(self.vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    args = parser.parse_args()
    compile_suffix(args.model_path, args.device, args.bundle)

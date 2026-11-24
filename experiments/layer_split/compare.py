"""Run the two stages in their installed Python environments over a private socketpair."""
import argparse
import json
import os
import socket
import subprocess
import sys
from multiprocessing.connection import Connection
from pathlib import Path

import torch


# Fixed before running either the split model or a logits comparison.
MAX_RMSE = 0.10
MAX_ABS = 0.50
DECODE_STEPS = 8


def suffix_worker(fd, model, bundle):
    from suffix import Suffix

    with Connection(fd) as channel, torch.inference_mode():
        suffix = Suffix(model, device=0, bundle=bundle)
        channel.send(os.getpid())
        while True:
            request = channel.recv()
            if request is None:
                return
            start, data = request
            hidden = torch.from_numpy(data)
            logits = suffix(hidden, start)
            if logits.ndim != 1 or logits.device.type != "cpu":
                raise RuntimeError(f"Unexpected suffix output: {logits.shape}, {logits.device}")
            channel.send(logits.float().contiguous().numpy())


def compare_step(logits, reference):
    if logits.shape != reference.shape or not torch.isfinite(reference).all():
        raise ValueError("Invalid reference or mismatched vocabulary")
    difference = logits - reference.float()
    maximum = difference.abs().max().item()
    rmse = difference.square().mean().sqrt().item()
    expected = reference.argmax().item()
    actual = logits.argmax().item()
    return dict(max_abs=maximum, rmse=rmse, gpu_top1=expected, split_top1=actual,
                passed=maximum <= MAX_ABS and rmse <= MAX_RMSE and actual == expected)


@torch.inference_mode()
def run(args):
    from gpu import Prefix
    from transformers import AutoTokenizer

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    golden = reference["logits"]
    if reference["dtype"] != "float16" or golden.dtype != torch.float16:
        raise ValueError("Expected FP16 full-GPU reference")
    prompt_ids = list(reference["prompt_ids"])
    tokens = list(reference["tokens"])
    config = json.loads((args.model / "config.json").read_text())
    if tuple(golden.shape) != (1 + DECODE_STEPS, config["vocab_size"]):
        raise ValueError(f"Unexpected golden logits: {tuple(golden.shape)}")
    if tokens != golden.argmax(dim=-1).tolist():
        raise ValueError("Reference tokens do not agree with the stored logits")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if prompt_ids != tokenizer.encode("The capital of France is", add_special_tokens=False):
        raise ValueError("Reference prompt/tokenizer mismatch")
    if Path(reference["model_path"]).resolve() != args.model.resolve():
        raise ValueError("Reference checkpoint mismatch")
    args.logs.mkdir(exist_ok=True, parents=True)
    limits = dict(rmse=MAX_RMSE, max_abs=MAX_ABS, require_top1=True)
    print(f"Criteria for every teacher-forced step: {limits}", flush=True)
    prefix = Prefix(str(args.model))
    addr, peer = socket.socketpair()
    with Connection(addr.detach()) as channel, peer, (args.logs / "npu.log").open("w") as log:
        proc = subprocess.Popen(
            [args.npu_python.absolute(), "-u", Path(__file__).absolute(),
             "--fd", str(peer.fileno()), "--model", str(args.model.resolve()),
             "--bundle", str(args.bundle.resolve())],
            cwd=args.logs, pass_fds=[peer.fileno()], stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "RBLN_DEVICES": "0", "RBLN_RUNTIME_TIMER": "1"},
        )
        peer.close()
        try:
            npu_pid = channel.recv()
            if npu_pid != proc.pid:
                raise RuntimeError("Unexpected NPU worker")
            metrics, generated = [], []
            for mode in ("teacher_forced", "greedy"):
                prefix.reset()
                ids, start = prompt_ids, 0
                for step in range(1 + DECODE_STEPS):
                    hidden = prefix.step(ids, start)
                    if (hidden.shape != (1, len(ids), config["hidden_size"])
                            or hidden.dtype != torch.float16 or hidden.device.type != "cpu"):
                        raise ValueError(f"Unexpected boundary: {hidden.shape} {hidden.dtype} {hidden.device}")
                    channel.send((start, hidden.contiguous().numpy()))
                    logits = torch.from_numpy(channel.recv())
                    if logits.shape != golden[step].shape or not torch.isfinite(logits).all():
                        raise ValueError(f"Invalid NPU logits at {mode} step {step}")
                    token = logits.argmax().item()
                    record = dict(run=mode, step=step, start=start, input_ids=ids, shape=list(hidden.shape))
                    if mode == "teacher_forced":
                        record.update(compare_step(logits, golden[step]))
                        metrics.append(record)
                    else:
                        record.update(split_top1=token)
                        generated.append(token)
                    print(json.dumps(record), flush=True)
                    if mode == "teacher_forced" and step == 1:
                        # Both processes still own their device allocations after a completed native decode.
                        with (args.logs / "devices_active.log").open("w") as out:
                            print(f"GPU PID {os.getpid()}, NPU PID {npu_pid}", file=out, flush=True)
                            subprocess.run(["nvidia-smi"], stdout=out, check=True)
                            subprocess.run(["rbln-stat"], stdout=out, check=True)
                    start += len(ids)
                    ids = [int(tokens[step]) if mode == "teacher_forced" else token]
            channel.send(None)
            if proc.wait() != 0:
                raise RuntimeError(f"NPU worker exit {proc.returncode}")
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            raise RuntimeError(f"NPU worker disconnected; see {log.name}") from exc
        finally:
            if proc.poll() is None:
                # Only this invocation's dedicated child can be stopped here.
                proc.terminate()
                proc.wait()
    result = dict(passed=all(row["passed"] for row in metrics), limits=limits,
                  model=str(args.model), prompt_ids=prompt_ids, steps=metrics,
                  greedy_ids=generated, greedy_text=tokenizer.decode(generated))
    print("Split greedy continuation:", repr(result["greedy_text"]), flush=True)
    (args.logs / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=Path, default=Path("logs/reference.pt"))
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent / "build/suffix")
    parser.add_argument("--npu-python", type=Path)
    args = parser.parse_args()
    if args.fd is not None:
        suffix_worker(args.fd, str(args.model), args.bundle)
    elif args.npu_python is None:
        parser.error("--npu-python required")
    else:
        sys.exit(run(args))

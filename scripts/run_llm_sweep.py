#!/usr/bin/env python3
"""Controlled residual-QJL sweeps for GPT-2 and OPT causal language models."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parse_specs(args) -> list[tuple[int, int, float]]:
    if args.spec_file:
        data = json.loads(args.spec_file.read_text())
        return [(int(r["key_bits"]), int(r["value_bits"]), float(r["m_over_d"])) for r in data]
    ratios = ([0.25] + args.ratios + [8.0]) if args.include_extremes else args.ratios
    return [(kb, vb, ratio) for conf in args.configs for kb, vb in [map(int, conf.split(","))] for ratio in ratios]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gpt2"])
    p.add_argument("--model-revision")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.5, 1, 2, 4])
    p.add_argument("--include-extremes", action="store_true")
    p.add_argument("--qjl-seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    p.add_argument("--configs", nargs="+", default=["4,2", "3,2"])
    p.add_argument("--spec-file", type=Path)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--token-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--study-id", default="exploratory")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/raw/llm_sweep.jsonl"))
    args = p.parse_args()
    if args.quick:
        args.models, args.tokens, args.qjl_seeds = args.models[:1], min(args.tokens, 128), args.qjl_seeds[:1]
    specs = parse_specs(args)

    try:
        import pandas as pd
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from qjlstudy.torch_quant import FastResidualQJLCache
    except Exception as exc:
        raise SystemExit(f"experiment dependencies unavailable: {exc}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dtype = torch.float16 if device.type == "mps" else torch.float32
    data_path = Path("data/wikitext-2-raw-v1-test.parquet")
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
            "wikitext-2-raw-v1/test-00000-of-00001.parquet",
            data_path,
        )
    text = "\n\n".join(pd.read_parquet(data_path)["text"].fillna(""))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def perplexity(model, input_ids):
        n = min(args.tokens, input_ids.shape[1] - 1)
        losses, evaluated = [], 0
        start_time = time.perf_counter()
        for start in range(0, n, args.stride):
            end = min(start + args.stride + 1, n + 1)
            chunk = input_ids[:, max(0, end - 1024):end]
            target = end - start - 1
            logits = model(chunk).logits[:, -target - 1:-1]
            labels = chunk[:, -target:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
            losses.append(loss.item() * target)
            evaluated += target
        return math.exp(sum(losses) / evaluated), time.perf_counter() - start_time

    def head_attention(cache, query, key, value, causal_mask, scale, layer, head, collector):
        query, key = query.float(), key.float()
        cache.encode(key, value)
        approximate = cache.scores(query)
        exact = query @ key.T
        reconstructed = cache.Knorm.unsqueeze(1) * (cache.K @ cache.rotation)
        residual_true = query @ (key - reconstructed).T
        residual_estimate = approximate - (query @ cache.rotation.T @ cache.K.T) * cache.Knorm
        exact_weights = torch.softmax(exact * scale + causal_mask, dim=1)
        approximate_weights = torch.softmax(approximate * scale + causal_mask, dim=1)
        valid = torch.ones_like(exact, dtype=torch.bool).tril()
        collector.append({
            "layer": layer,
            "head": head,
            "attention_logit_mae": ((approximate - exact) * scale)[valid].abs().mean().item(),
            "attention_logit_rmse": ((approximate - exact) * scale)[valid].square().mean().sqrt().item(),
            "qjl_residual_rmse": ((residual_estimate - residual_true) * scale)[valid].square().mean().sqrt().item(),
            "attention_kl": (exact_weights * (exact_weights.clamp_min(1e-12).log() - approximate_weights.clamp_min(1e-12).log())).sum(dim=1).mean().item(),
            "residual_norm_mean": cache.Rnorm.mean().item(),
            "residual_norm_std": cache.Rnorm.std().item(),
            "key_norm_mean": cache.Knorm.mean().item(),
        })
        return approximate_weights @ cache.values()

    def replace_gpt2(attn, layer, key_bits, value_bits, m, seed, collector):
        heads, d = attn.num_heads, attn.head_dim
        caches = [FastResidualQJLCache(d, key_bits, value_bits, layer, h, m, seed, device) for h in range(heads)]

        def forward(hidden_states, **kwargs):
            batch, seq, _ = hidden_states.shape
            q, k, v = attn.c_attn(hidden_states).split(attn.split_size, dim=2)
            shape = (batch, seq, heads, d)
            q, k, v = (x.view(shape).transpose(1, 2) for x in (q, k, v))
            mask = torch.triu(torch.full((seq, seq), float("-inf"), device=device), 1)
            output = [head_attention(caches[h], q[0, h], k[0, h], v[0, h], mask, 1 / math.sqrt(d), layer, h, collector) for h in range(heads)]
            joined = torch.stack(output).unsqueeze(0).transpose(1, 2).reshape(batch, seq, heads * d)
            return attn.resid_dropout(attn.c_proj(joined.to(hidden_states.dtype))), None

        attn.forward = forward

    def replace_opt(attn, layer, key_bits, value_bits, m, seed, collector):
        heads, d = attn.num_heads, attn.head_dim
        caches = [FastResidualQJLCache(d, key_bits, value_bits, layer, h, m, seed, device) for h in range(heads)]

        def forward(hidden_states, **kwargs):
            batch, seq, _ = hidden_states.shape
            shape = (batch, seq, heads, d)
            q = attn.q_proj(hidden_states).view(shape).transpose(1, 2)
            k = attn.k_proj(hidden_states).view(shape).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(shape).transpose(1, 2)
            mask = torch.triu(torch.full((seq, seq), float("-inf"), device=device), 1)
            output = [head_attention(caches[h], q[0, h], k[0, h], v[0, h], mask, 1 / math.sqrt(d), layer, h, collector) for h in range(heads)]
            joined = torch.stack(output).unsqueeze(0).transpose(1, 2).reshape(batch, seq, heads * d)
            return attn.out_proj(joined.to(hidden_states.dtype)), None

        attn.forward = forward

    def architecture(model):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return "gpt2", [(i, block.attn) for i, block in enumerate(model.transformer.h)]
        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            return "opt", [(i, block.self_attn) for i, block in enumerate(model.model.decoder.layers)]
        raise ValueError(f"unsupported architecture: {model.__class__.__name__}")

    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=args.model_revision)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.token_offset + args.tokens + 1,
        ).input_ids
        input_ids = encoded[:, args.token_offset:args.token_offset + args.tokens + 1].to(device)
        load_kwargs = dict(dtype=model_dtype, attn_implementation="eager", revision=args.model_revision)
        baseline = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device).eval()
        family, baseline_layers = architecture(baseline)
        fp_ppl, fp_runtime = perplexity(baseline, input_ids)
        num_layers = len(baseline_layers)
        num_heads, d = baseline_layers[0][1].num_heads, baseline_layers[0][1].head_dim
        del baseline

        for key_bits, value_bits, ratio in specs:
            m = max(1, round(ratio * d))
            for seed in args.qjl_seeds:
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device).eval()
                _, layers = architecture(model)
                diagnostics = []
                for layer, attn in layers:
                    replacer = replace_gpt2 if family == "gpt2" else replace_opt
                    replacer(attn, layer, key_bits, value_bits, m, seed, diagnostics)
                ppl, runtime = perplexity(model, input_ids)
                layer_metrics = []
                for layer in range(num_layers):
                    records = [r for r in diagnostics if r["layer"] == layer]
                    layer_metrics.append({"layer": layer, **{
                        metric: sum(r[metric] for r in records) / len(records)
                        for metric in ("attention_logit_rmse", "qjl_residual_rmse", "attention_kl", "residual_norm_mean", "residual_norm_std", "key_norm_mean")
                    }})
                metric = lambda name: sum(r[name] for r in diagnostics) / len(diagnostics)
                record = {
                    "study_id": args.study_id,
                    "model": model_name,
                    "architecture": family,
                    "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
                    "token_offset": args.token_offset,
                    "tokens": args.tokens,
                    "context_length": 1024,
                    "model_dtype": str(model_dtype).replace("torch.", ""),
                    "device": device.type,
                    "key_bits": key_bits,
                    "value_bits": value_bits,
                    "d": d,
                    "m": m,
                    "m_over_d": m / d,
                    "qjl_seed": seed,
                    "perplexity": ppl,
                    "ppl_delta_fp": ppl - fp_ppl,
                    "fp16_perplexity": fp_ppl,
                    "fp16_runtime_s": fp_runtime,
                    "runtime_s": runtime,
                    "attention_logit_mae": metric("attention_logit_mae"),
                    "attention_logit_rmse": metric("attention_logit_rmse"),
                    "qjl_residual_rmse": metric("qjl_residual_rmse"),
                    "attention_kl_fp_to_quantized": metric("attention_kl"),
                    "residual_norm_mean": metric("residual_norm_mean"),
                    "residual_norm_std": metric("residual_norm_std"),
                    "key_norm_mean": metric("key_norm_mean"),
                    "layer_metrics": layer_metrics,
                    "qjl_sketch_bytes_per_key": math.ceil(m / 8) + 4,
                    "key_storage_bytes_per_key": math.ceil(d * (key_bits - 1) / 8) + math.ceil(m / 8) + 8,
                    "value_storage_bytes_per_value": math.ceil(d * value_bits / 8) + 4,
                    "kv_storage_bytes_per_token": math.ceil(d * (key_bits - 1) / 8) + math.ceil(m / 8) + 8 + math.ceil(d * value_bits / 8) + 4,
                    "shared_projection_bytes_per_layer_head": m * d * 4,
                    "shared_projection_bytes_total": num_layers * num_heads * m * d * 4,
                    "git_commit": git_hash(),
                }
                with args.out.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                keys = ("study_id", "model", "key_bits", "value_bits", "m_over_d", "qjl_seed", "perplexity", "attention_logit_rmse", "kv_storage_bytes_per_token")
                print(json.dumps({k: record[k] for k in keys}))


if __name__ == "__main__":
    main()

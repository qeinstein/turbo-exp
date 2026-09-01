#!/usr/bin/env python3
"""Controlled GPT-2-style QJL m sweep; writes one JSON object per configuration."""
from __future__ import annotations
import argparse, csv, json, math, subprocess, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['gpt2', 'distilgpt2'])
    p.add_argument('--ratios', type=float, nargs='+', default=[.5,1,2,4])
    p.add_argument('--include-extremes', action='store_true')
    p.add_argument('--qjl-seeds', type=int, nargs='+', default=[11,23,37,53,71])
    p.add_argument('--configs', nargs='+', default=['4,2','3,2'])
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--out', type=Path, default=Path('results/raw/llm_sweep.jsonl'))
    a=p.parse_args()
    if a.quick: a.models, a.tokens, a.qjl_seeds = a.models[:1], min(a.tokens, 256), a.qjl_seeds[:1]
    if a.include_extremes: a.ratios=[.25]+a.ratios+[8]
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from qjlstudy.torch_quant import FastResidualQJLCache
    except Exception as e:
        raise SystemExit(f'experiment dependencies unavailable: {e}')
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    text='\n\n'.join(load_dataset('wikitext','wikitext-2-raw-v1',split='test')['text'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    def ppl(model,tok):
        ids=tok(text,return_tensors='pt').input_ids.to(device); n=min(a.tokens,ids.shape[1]-1); losses=[]; t=time.perf_counter()
        for start in range(0,n,a.stride):
            x=ids[:,max(0,start-1024+a.stride):min(start+a.stride,ids.shape[1])]; target=min(start+a.stride,ids.shape[1])-start
            o=model(x,labels=x); logits=o.logits[:,-target:-1]; labels=x[:,-target+1:]
            losses.append(torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1)).item()*target)
        return math.exp(sum(losses)/n), time.perf_counter()-t
    # GPT-2 eager attention replacement, deliberately narrow rather than claiming cross-architecture support.
    def replace(attn,kb,vb,m,seed,collector):
        heads,d=attn.num_heads,attn.head_dim; caches=[FastResidualQJLCache(d,kb,vb,l,h,m,seed,device) for h in range(heads)]
        original=attn.forward
        def forward(hidden_states, **kwargs):
            b,s,_=hidden_states.shape; q,k,v=attn.c_attn(hidden_states).split(attn.split_size,dim=2); shape=(b,s,heads,d)
            Q=q.view(shape).transpose(1,2); K=k.view(shape).transpose(1,2); V=v.view(shape).transpose(1,2); mask=torch.triu(torch.full((s,s),float('-inf'),device=device),1)
            out=[]
            for h,c in enumerate(caches):
                c.encode(K[0,h],V[0,h]); approx=c.scores(Q[0,h]); fp=Q[0,h]@K[0,h].T
                residual_true=Q[0,h] @ (K[0,h] - (c.Knorm.unsqueeze(1) * (c.K @ c.rotation))).T
                residual_est=approx - (Q[0,h] @ c.rotation.T @ c.K.T) * c.Knorm
                fp_w=torch.softmax((fp+mask)/math.sqrt(d),dim=1)
                w=torch.softmax((approx+mask)/math.sqrt(d),dim=1)
                collector.extend([(approx-fp).abs().mean().item(), ((approx-fp).square().mean().sqrt().item()),
                                  (residual_est-residual_true).square().mean().sqrt().item(),
                                  (fp_w*(fp_w.clamp_min(1e-12).log()-w.clamp_min(1e-12).log())).sum(dim=1).mean().item()])
                out.append(w@c.values())
            z=torch.stack(out).unsqueeze(0).transpose(1,2).reshape(b,s,heads*d); return attn.c_proj(attn.resid_dropout(z)),None
        attn.forward=forward
    for name in a.models:
      tok=AutoTokenizer.from_pretrained(name); base=AutoModelForCausalLM.from_pretrained(name).to(device).eval(); fp,fpsec=ppl(base,tok); del base
      for conf in a.configs:
       kb,vb=map(int,conf.split(','))
       for ratio in a.ratios:
        for seed in a.qjl_seeds:
         model=AutoModelForCausalLM.from_pretrained(name).to(device).eval(); d=model.transformer.h[0].attn.head_dim; m=max(1,round(ratio*d)); errors=[]
         for l,block in enumerate(model.transformer.h): replace(block.attn,kb,vb,m,seed,errors)
         pval,secs=ppl(model,tok); rec={'model':name,'dataset':'wikitext-2-raw-v1:test','tokens':a.tokens,'context_length':1024,'key_bits':kb,'value_bits':vb,'d':d,'m':m,'m_over_d':m/d,'qjl_seed':seed,'perplexity':pval,'ppl_delta_fp':pval-fp,'fp16_perplexity':fp,'fp16_runtime_s':fpsec,'runtime_s':secs,'attention_logit_mae':sum(errors[::4])/len(errors[::4]),'attention_logit_rmse':sum(errors[1::4])/len(errors[1::4]),'qjl_residual_rmse':sum(errors[2::4])/len(errors[2::4]),'attention_kl_fp_to_quantized':sum(errors[3::4])/len(errors[3::4]),'qjl_sketch_bytes_per_key':math.ceil(m/8)+4,'key_storage_bytes_per_key':math.ceil(d*(kb-1)/8)+math.ceil(m/8)+8,'value_storage_bytes_per_value':math.ceil(d*vb/8)+4,'git_commit':git_hash()}
         with a.out.open('a') as f: f.write(json.dumps(rec)+'\n')
         print(json.dumps(rec))

if __name__=='__main__': main()

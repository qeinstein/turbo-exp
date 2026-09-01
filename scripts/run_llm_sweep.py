#!/usr/bin/env python3
"""Controlled GPT-2-style QJL m sweep; writes one JSON object per configuration."""
from __future__ import annotations
import argparse, json, math, subprocess, time, urllib.request
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
        import pandas as pd
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from qjlstudy.torch_quant import FastResidualQJLCache
    except Exception as e:
        raise SystemExit(f'experiment dependencies unavailable: {e}')
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_dtype=torch.float16 if device.type == 'mps' else torch.float32
    # Direct immutable-file route avoids a datasets 5.x resolver hang observed
    # on this host. The file is the official WikiText-2 raw test parquet shard.
    data_path=Path('data/wikitext-2-raw-v1-test.parquet')
    if not data_path.exists():
        data_path.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve('https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet',data_path)
    text='\n\n'.join(pd.read_parquet(data_path)['text'].fillna(''))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    @torch.inference_mode()
    def ppl(model,ids):
        n=min(a.tokens,ids.shape[1]-1); losses=[]; evaluated=0; t=time.perf_counter()
        for start in range(0,n,a.stride):
            end=min(start+a.stride+1,n+1); x=ids[:,max(0,end-1024):end]; target=end-start-1
            o=model(x); logits=o.logits[:,-target-1:-1]; labels=x[:,-target:]
            losses.append(torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1)).item()*target); evaluated+=target
        return math.exp(sum(losses)/evaluated), time.perf_counter()-t
    # GPT-2 eager attention replacement, deliberately narrow rather than claiming cross-architecture support.
    def replace(attn,kb,vb,m,seed,collector):
        heads,d=attn.num_heads,attn.head_dim; caches=[FastResidualQJLCache(d,kb,vb,l,h,m,seed,device) for h in range(heads)]
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
                valid=torch.ones((s,s),device=device,dtype=torch.bool).tril()
                collector.extend([(approx-fp)[valid].abs().mean().item(), ((approx-fp)[valid].square().mean().sqrt().item()),
                                  (residual_est-residual_true)[valid].square().mean().sqrt().item(),
                                  (fp_w*(fp_w.clamp_min(1e-12).log()-w.clamp_min(1e-12).log())).sum(dim=1).mean().item()])
                out.append(w@c.values())
            z=torch.stack(out).unsqueeze(0).transpose(1,2).reshape(b,s,heads*d)
            z=attn.c_proj(z.to(hidden_states.dtype)); return attn.resid_dropout(z),None
        attn.forward=forward
    for name in a.models:
      tok=AutoTokenizer.from_pretrained(name); ids=tok(text,return_tensors='pt').input_ids[:,:a.tokens+1].to(device)
      base=AutoModelForCausalLM.from_pretrained(name,dtype=model_dtype,attn_implementation='eager').to(device).eval(); fp,fpsec=ppl(base,ids); del base
      for conf in a.configs:
       kb,vb=map(int,conf.split(','))
       for ratio in a.ratios:
        for seed in a.qjl_seeds:
         model=AutoModelForCausalLM.from_pretrained(name,dtype=model_dtype,attn_implementation='eager').to(device).eval(); d=model.transformer.h[0].attn.head_dim; m=max(1,round(ratio*d)); errors=[]
         for l,block in enumerate(model.transformer.h): replace(block.attn,kb,vb,m,seed,errors)
         pval,secs=ppl(model,ids); rec={'model':name,'dataset':'Salesforce/wikitext:wikitext-2-raw-v1:test','tokens':a.tokens,'context_length':1024,'model_dtype':str(model_dtype).replace('torch.',''),'device':device.type,'key_bits':kb,'value_bits':vb,'d':d,'m':m,'m_over_d':m/d,'qjl_seed':seed,'perplexity':pval,'ppl_delta_fp':pval-fp,'fp16_perplexity':fp,'fp16_runtime_s':fpsec,'runtime_s':secs,'attention_logit_mae':sum(errors[::4])/len(errors[::4]),'attention_logit_rmse':sum(errors[1::4])/len(errors[1::4]),'qjl_residual_rmse':sum(errors[2::4])/len(errors[2::4]),'attention_kl_fp_to_quantized':sum(errors[3::4])/len(errors[3::4]),'qjl_sketch_bytes_per_key':math.ceil(m/8)+4,'key_storage_bytes_per_key':math.ceil(d*(kb-1)/8)+math.ceil(m/8)+8,'value_storage_bytes_per_value':math.ceil(d*vb/8)+4,'git_commit':git_hash()}
         with a.out.open('a') as f: f.write(json.dumps(rec)+'\n')
         print(json.dumps(rec))

if __name__=='__main__': main()

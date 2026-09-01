#!/usr/bin/env python3
"""Warm-cache microbenchmark for the exact residual-QJL attention operations."""
from __future__ import annotations
import argparse, json, random, statistics, time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from qjlstudy.torch_quant import FastResidualQJLCache

def sync(device):
    if device.type == 'mps': torch.mps.synchronize()
    elif device.type == 'cuda': torch.cuda.synchronize()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--d',type=int,default=64); p.add_argument('--tokens',type=int,default=512)
    p.add_argument('--ratios',type=float,nargs='+',default=[.5,1,2,4]); p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37,53,71])
    p.add_argument('--repeats',type=int,default=20); p.add_argument('--out',type=Path,default=Path('results/raw/cache_runtime.json'))
    a=p.parse_args(); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); g=torch.Generator(device=device); g.manual_seed(991)
    q=torch.randn(a.tokens,a.d,generator=g,device=device); k=torch.randn(a.tokens,a.d,generator=g,device=device); v=torch.randn(a.tokens,a.d,generator=g,device=device)
    jobs=[(ratio,seed) for ratio in a.ratios for seed in a.seeds]; random.Random(17).shuffle(jobs); rows=[]
    with torch.inference_mode():
      for ratio,seed in jobs:
        m=round(a.d*ratio); cache=FastResidualQJLCache(a.d,4,2,0,0,m,seed,device)
        for _ in range(3): cache.encode(k,v); cache.scores(q); cache.values()
        sync(device); timings=[]
        for _ in range(a.repeats):
            start=time.perf_counter(); cache.encode(k,v); cache.scores(q); cache.values(); sync(device); timings.append(time.perf_counter()-start)
        rows.append({'d':a.d,'m':m,'m_over_d':ratio,'qjl_seed':seed,'tokens':a.tokens,'repeats':a.repeats,'device':device.type,
                     'runtime_s_mean':statistics.mean(timings),'runtime_s_std':statistics.stdev(timings),'runtime_s_median':statistics.median(timings),
                     'measurement_bytes_per_key':(m+7)//8,'shared_projection_bytes':m*a.d*4})
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(rows,indent=2)+'\n')
    for ratio in a.ratios:
        vals=[r['runtime_s_median'] for r in rows if r['m_over_d']==ratio]
        print(f'm/d={ratio:g}: median across seed medians={statistics.median(vals):.6f}s')
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Write preregistered exact-storage allocation configurations."""
from __future__ import annotations
import argparse, json
from itertools import product
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qjlstudy.budget import enumerate_matched_configs, storage_bytes


def comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(','))


def comma_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(','))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--d',type=int,default=64); p.add_argument('--out',type=Path,default=Path('configs/phase2_equal_budget_full.json')); p.add_argument('--exclude-results',type=Path)
    p.add_argument('--key-bits',type=comma_ints,default=(2,3,4,5)); p.add_argument('--value-bits',type=comma_ints,default=(1,2,3,4)); p.add_argument('--ratios',type=comma_floats,default=(0,0.5,1,1.5,2,3,4)); p.add_argument('--all-configs',action='store_true'); a=p.parse_args()
    if a.all_configs:
        configs=[]
        for key_bits,value_bits,ratio in product(a.key_bits,a.value_bits,a.ratios):
            m=round(a.d*ratio)
            configs.append({'key_bits':key_bits,'value_bits':value_bits,'m':m,'m_over_d':m/a.d,**storage_bytes(a.d,key_bits,value_bits,m)})
        configs.sort(key=lambda r:(r['kv_bytes'],r['key_bits'],r['value_bits'],r['m']))
    else:
        configs=enumerate_matched_configs(a.d,key_bits=a.key_bits,value_bits=a.value_bits,ratios=a.ratios)
    if a.exclude_results:
        existing={(r['key_bits'],r['value_bits'],r['m_over_d']) for line in a.exclude_results.read_text().splitlines() if line.strip() for r in [json.loads(line)]}
        configs=[r for r in configs if (r['key_bits'],r['value_bits'],r['m_over_d']) not in existing]
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(configs,indent=2)+'\n')
    groups={r['kv_bytes'] for r in configs}; print(f'wrote {len(configs)} configurations in {len(groups)} budget groups: {sorted(groups)}')
if __name__=='__main__': main()

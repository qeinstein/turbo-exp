#!/usr/bin/env python3
"""Write preregistered exact-storage allocation configurations."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qjlstudy.budget import enumerate_matched_configs

def main():
    p=argparse.ArgumentParser(); p.add_argument('--d',type=int,default=64); p.add_argument('--out',type=Path,default=Path('configs/phase2_equal_budget_full.json')); p.add_argument('--exclude-results',type=Path); a=p.parse_args()
    configs=enumerate_matched_configs(a.d)
    if a.exclude_results:
        existing={(r['key_bits'],r['value_bits'],r['m_over_d']) for line in a.exclude_results.read_text().splitlines() if line.strip() for r in [json.loads(line)]}
        configs=[r for r in configs if (r['key_bits'],r['value_bits'],r['m_over_d']) not in existing]
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(configs,indent=2)+'\n')
    groups={r['kv_bytes'] for r in configs}; print(f'wrote {len(configs)} configurations in {len(groups)} matched budget groups: {sorted(groups)}')
if __name__=='__main__': main()

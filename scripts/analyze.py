#!/usr/bin/env python3
"""Generate tables and falsification plots solely from raw JSONL experiment records."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RAW=Path('results/raw/llm_sweep.jsonl'); OUT=Path('results/summary'); PLOTS=Path('results/plots')
def main():
    if not RAW.exists(): raise SystemExit(f'no raw results: run scripts/run_llm_sweep.py first ({RAW})')
    rows=[json.loads(x) for x in RAW.read_text().splitlines() if x.strip()]
    groups={}
    for r in rows: groups.setdefault(tuple(r[k] for k in ('model','key_bits','value_bits','m_over_d')),[]).append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        rec=dict(zip(('model','key_bits','value_bits','m_over_d'),key)); rec['n_seeds']=len(rs); rec['m']=rs[0]['m']
        for metric in ('perplexity','ppl_delta_fp','attention_logit_mae','attention_logit_rmse','qjl_residual_rmse','attention_kl_fp_to_quantized','runtime_s','qjl_sketch_bytes_per_key','key_storage_bytes_per_key'):
            vals=np.array([r[metric] for r in rs]); rec[metric+'_mean']=float(vals.mean()); rec[metric+'_std']=float(vals.std(ddof=1)) if len(vals)>1 else 0.
        out.append(rec)
    for label in sorted(set((r['model'],r['key_bits'],r['value_bits']) for r in out)):
        series=sorted((r for r in out if (r['model'],r['key_bits'],r['value_bits'])==label),key=lambda r:r['m_over_d'])
        for previous,current in zip(series,series[1:]):
            added=current['qjl_sketch_bytes_per_key_mean']-previous['qjl_sketch_bytes_per_key_mean']
            gain=previous['perplexity_mean']-current['perplexity_mean']
            current['previous_m_over_d']=previous['m_over_d']; current['added_qjl_bytes']=added
            current['perplexity_gain_from_previous']=gain; current['perplexity_gain_per_added_qjl_byte']=gain/added
        series[0].update(previous_m_over_d='',added_qjl_bytes='',perplexity_gain_from_previous='',perplexity_gain_per_added_qjl_byte='')
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    cols=list(out[0]); (OUT/'summary.csv').write_text(','.join(cols)+'\n'+'\n'.join(','.join(str(r[c]) for c in cols) for r in out)+'\n')
    try:
        import matplotlib.pyplot as plt
    except ImportError: print('summary written; matplotlib unavailable'); return
    PLOTS.mkdir(parents=True,exist_ok=True)
    for metric,title,name in [('perplexity_mean','Perplexity','perplexity_vs_m'),('attention_logit_rmse_mean','Attention-logit RMSE','logit_error_vs_m'),('qjl_residual_rmse_mean','QJL residual RMSE','qjl_error_vs_m'),('key_storage_bytes_per_key_mean','Key bytes per token','storage_vs_m')]:
        fig,ax=plt.subplots();
        for label in sorted(set((r['model'],r['key_bits'],r['value_bits']) for r in out)):
            rs=[r for r in out if (r['model'],r['key_bits'],r['value_bits'])==label]; rs.sort(key=lambda r:r['m_over_d'])
            ax.errorbar([r['m_over_d'] for r in rs],[r[metric] for r in rs],yerr=[r.get(metric.replace('_mean','_std'),0) for r in rs],marker='o',label=f'{label[0]} {label[1]}/{label[2]}b')
        ax.set_xscale('log',base=2); ax.set_xlabel('m/d'); ax.set_ylabel(title); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS/(name+'.png'),dpi=160); plt.close(fig)
    fig,ax=plt.subplots()
    for label in sorted(set((r['model'],r['key_bits'],r['value_bits']) for r in out)):
        rs=sorted((r for r in out if (r['model'],r['key_bits'],r['value_bits'])==label),key=lambda r:r['qjl_sketch_bytes_per_key_mean'])
        ax.plot([r['qjl_sketch_bytes_per_key_mean'] for r in rs],[r['perplexity_mean'] for r in rs],marker='o',label=f'{label[0]} {label[1]}/{label[2]}b')
    ax.set_xlabel('QJL bytes per key (packed signs + residual norm)'); ax.set_ylabel('Perplexity'); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS/'quality_vs_qjl_storage.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots()
    for label in sorted(set((r['model'],r['key_bits'],r['value_bits']) for r in rows)):
        rs=[r for r in rows if (r['model'],r['key_bits'],r['value_bits'])==label]
        ax.scatter([r['m_over_d'] for r in rs],[r['perplexity'] for r in rs],alpha=.75,label=f'{label[0]} {label[1]}/{label[2]}b')
    ax.set_xscale('log',base=2); ax.set_xlabel('m/d'); ax.set_ylabel('Individual-seed perplexity'); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS/'seed_variance.png',dpi=160); plt.close(fig)
    print(f'wrote {len(out)} summary groups to {OUT}')
if __name__=='__main__': main()

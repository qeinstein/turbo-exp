#!/usr/bin/env python3
"""Capture immutable model/data checksums and software versions for the study."""
from __future__ import annotations
import hashlib, importlib.metadata, json, platform, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

hub=Path.home()/'.cache/huggingface/hub/models--gpt2'; revision=(hub/'refs/main').read_text().strip(); snapshot=hub/'snapshots'/revision
weight=(snapshot/'model.safetensors').resolve(); data=ROOT/'data/wikitext-2-raw-v1-test.parquet'
record={'captured_from_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'python':platform.python_version(),'platform':platform.platform(),
        'model':'gpt2','model_revision':revision,'model_weight_sha256':sha256(weight),'dataset':'Salesforce/wikitext:wikitext-2-raw-v1:test',
        'dataset_url':'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet','dataset_sha256':sha256(data),
        'packages':{name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','scipy','pandas','matplotlib')}}
out=ROOT/'results/raw/environment.json'; out.write_text(json.dumps(record,indent=2)+'\n'); print(json.dumps(record,indent=2))

"""Same fixed v37 inputs; read only, no feature copies or persistent caches."""
import argparse
from pathlib import Path
import numpy as np
from fs_data import validate_splits,FSData,labels
from spec import HERE,TASKS,sha

def check(root):
    root=Path(root).resolve();validate_splits(root)
    entries=(HERE/'FS1000_INPUT_SHA256SUMS').read_text().splitlines()
    if len(entries)!=3743:raise ValueError('Expected 3743 required input hashes')
    for i,line in enumerate(entries,1):
        expected,name=line.split('  ',1)
        file=root/name;file.resolve().relative_to(root)
        if sha(file)!=expected:raise ValueError('Input hash mismatch: '+str(file))
        if i%100==0 or i==len(entries):print('Input hashes %d/%d PASS'%(i,len(entries)),flush=True)
    for train in (True,False):
        ds=FSData(root,'TES',train,external=True)
        for i in range(len(ds)):ds.features(i)
        ds.features.cache_clear()
    for task in TASKS:
        rows=labels(root/'train_fs1000_new.txt',task);labels(root/'val_fs1000_new.txt',task)
        if not np.all(np.diff(np.quantile([r[1] for r in rows],[.25,.5,.75]))>0):raise ValueError('Non-distinct training quartiles: '+task)
    print('FS1000_INPUTS_PASS: official split, labels, headers, finite values, alignment and all hashes',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);check(p.parse_args().data)

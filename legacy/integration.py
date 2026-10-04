"""Actual six-configuration two-epoch synthetic CPU integration, never research scores."""
import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader,Dataset
import adapter as a
import worker as w
import spec as c

class Synthetic(Dataset):
    def __init__(self):self.rows=[('synthetic_%d'%i,.2+i*.075) for i in range(8)]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        gen=torch.Generator().manual_seed(101+i)
        xs=tuple(torch.randn(12,d,generator=gen) for d in (768,768,1024))
        return xs+(torch.tensor(self.rows[i][1],dtype=torch.float32),)

def complete(root,method,out):
    files={p.name:c.sha(p) for p in out.iterdir() if p.is_file()}
    c.save(c.cell_path(root,'smoke','TES',0,method)/'complete.json',dict(attempt=out.name,files=files))

def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='v38_synthetic_') as tmp:
        root=Path(tmp);ds=Synthetic();train=DataLoader(ds,batch_size=4,shuffle=True,num_workers=0);test=DataLoader(ds,batch_size=1)
        a.seed_all(0);native=a.build_native(True)
        from train import train_epoch
        from models.loss import LossFun
        loss=LossFun(1.,1.);opt=torch.optim.Adam(native.parameters(),lr=1e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2,eta_min=1e-6)
        ref=root/'old_v37';ref.mkdir();best=-float('inf');selected=None
        for epoch in range(2):
            train_epoch(epoch,native,loss,train,opt,None,torch.device('cpu'),SimpleNamespace(score_range=130.))
            scheduler.step();replay=a.capture_rng();result=w.native_evaluate(native,test,torch.device('cpu'))
            if result['coef'] is not None and result['coef']>best:
                best=result['coef'];selected=epoch+1
                a.tensor_save(ref/'model.pth',native.state_dict());a.tensor_save(ref/'model.pth.eval_rng.pt',replay)
                c.save(ref/'selected_result.json',result)
        native.load_state_dict(a.load(ref/'model.pth'),strict=True)
        with a.preserved_rng(a.load(ref/'model.pth.eval_rng.pt')):result=w.native_evaluate(native,test,torch.device('cpu'))
        w.parity(result,c.read(ref/'selected_result.json'))
        result.update(task='TES',seed=0,method='mcmoe',phase='smoke',epochs=2,selected_epoch=selected,
                      reloaded_exact=True,ids=[r[0] for r in ds.rows],dependencies={})
        c.save(ref/'test_result.json',result)
        record=dict(directory=str(ref),required_files={n:c.sha(ref/n) for n in ('model.pth','model.pth.eval_rng.pt','test_result.json')})
        c.save(c.reference_path(root,'smoke','TES',0)/'record.json',record)
        c.save(c.reference_path(root,'smoke','TES',0)/'test_result.json',result)
        original=dict(record['required_files']);del native,opt
        for method in c.NEW_METHODS:
            out=c.cell_path(root,'smoke','TES',0,method)/'attempt_test';out.mkdir(parents=True)
            args=argparse.Namespace(root=str(root),phase='smoke',task='TES',seed=0,method=method,out=str(out),data='',cpu=True,workers=0)
            result=w.execute(args,train,test)
            assert result['reloaded_exact'] and result['epochs']==2 and result['synthetic_cpu_test_only']
            if method!='mc_ordered':
                check=c.read(out/'qmd_training.json');assert check['anchor_unchanged'] and check['statistics_unchanged'] and check['branch_updated']
                mech=c.read(out/'mechanism.json');w.parity(mech['variants']['full'],result)
            complete(root,method,out)
        assert original=={n:c.sha(ref/n) for n in original},'v37 reference was modified'
        print('V38_SIX_CONFIGURATION_CPU_INTEGRATION_PASS; synthetic only, no experimental claims',flush=True)

if __name__=='__main__':main()

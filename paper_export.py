"""Verified tables, controls and case queue; never turn completion into success."""
import csv
import json
from pathlib import Path
import shutil
import bootstrap
import numpy as np
from scipy.stats import rankdata, spearmanr
from contract import HERE,POLICY,TASKS,SEEDS,METHODS,cell,reference,read,save,sha,completed
from spec import SCORED

ALL=tuple(SCORED)+METHODS
NAMES=dict(mcmoe='MCMoE',mc_ordered='MCMoE + Ordered',mc_e='MCMoE + E',
    mc_u='MCMoE + E + U',mc_ordered_e='MCMoE + Ordered + E',
    mc_ordered_u='MCMoE + Ordered + E + U',mc_plain_e='MCMoE + plain residual',
    mc_fused_u='MCMoE + E + fused U')

def csv_write(path,rows):
    rows=list(rows)
    if not rows:raise ValueError('Empty table '+str(path))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def check_prediction(r,n=247):
    p=np.asarray(r['predictions']['prediction'],dtype=float)
    y=np.asarray(r['predictions']['label'],dtype=float)
    if p.shape!=(n,) or y.shape!=(n,) or len(r['ids'])!=n or len(set(r['ids']))!=n:
        raise ValueError('Missing/duplicate samples')
    if not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Nonfinite prediction')
    coef=float(spearmanr(p,y)[0]*100)
    if not np.isfinite(coef) or abs(coef-r['coef'])>1e-8:raise ValueError('Saved SRCC differs from raw predictions')
    return p,y,coef

def summarize(records):
    rows=[];seed_rows=[];task_rows=[]
    for m in ALL:
        values=[]
        for s in SEEDS:
            mean=float(np.mean([records[t,s,m]['coef'] for t in TASKS]));values.append(mean)
            base=float(np.mean([records[t,s,'mcmoe']['coef'] for t in TASKS]))
            seed_rows.append(dict(method=m,seed=s,**{t:records[t,s,m]['coef'] for t in TASKS},macro=mean,delta_vs_mcmoe=mean-base))
        for t in TASKS:task_rows.append(dict(method=m,task=t,mean=float(np.mean([records[t,s,m]['coef'] for s in SEEDS]))))
        rows.append(dict(method=m,name=NAMES[m],mean=float(np.mean(values)),seed_sd=float(np.std(values,ddof=1)),
            delta_vs_mcmoe=float(np.mean(values)-np.mean([records[t,s,'mcmoe']['coef'] for t in TASKS for s in SEEDS]))))
    pairs=[]
    for control,proposed in POLICY['comparison_pairs'].items():
        ds=[records[t,s,proposed]['coef']-records[t,s,control]['coef'] for t in TASKS for s in SEEDS]
        seed=[float(np.mean([records[t,s,proposed]['coef']-records[t,s,control]['coef'] for t in TASKS])) for s in SEEDS]
        pairs.append(dict(proposed=proposed,control=control,mean_gain=float(np.mean(ds)),seed_gains=seed,
            positive_seed_means=sum(x>0 for x in seed),wins=sum(x>1e-10 for x in ds),ties=sum(abs(x)<=1e-10 for x in ds),
            losses=sum(x < -1e-10 for x in ds),interpretation='supports mean advantage' if np.mean(ds)>0 else 'does not support mean advantage'))
    return rows,seed_rows,task_rows,pairs

def audit_cost(r):
    cost=r['cost'];rows=cost['variants']
    expected=('legacy_anchor','legacy_e','legacy_u','deploy_anchor','deploy_e','deploy_u')
    if cost['status']!='measured' or set(rows)!=set(expected):raise ValueError('Missing cost variants')
    for name,v in rows.items():
        for metric in ('wall_ms','cuda_ms'):
            x=np.asarray(v[metric],dtype=float)
            if x.shape!=(60,) or not np.isfinite(x).all() or (x<=0).any():raise ValueError('Incomplete/invalid timing')
            if abs(float(np.median(x))-v[metric+'_median'])>1e-8:raise ValueError('Cost median mismatch')
    return dict(task=r['task'],seed=r['seed'],parent=r['parent'],exclusive=cost['observed_exclusive'],
        device=cost['device'],setup_ms=r['session_setup_wall_ms'],
        **{m+'_'+k:rows[m][k+'_median'] for m in expected for k in ('wall_ms','cuda_ms')},
        deploy_u_overhead_pct=(rows['deploy_u']['wall_ms_median']/rows['deploy_anchor']['wall_ms_median']-1)*100,
        legacy_u_overhead_pct=(rows['legacy_u']['wall_ms_median']/rows['legacy_anchor']['wall_ms_median']-1)*100,
        deployment_speedup_pct=(1-rows['deploy_u']['wall_ms_median']/rows['legacy_u']['wall_ms_median'])*100,
        parameters_anchor=r['parameters']['anchor'],parameters_e=r['parameters']['e'],parameters_u=r['parameters']['u'],
        batch1_vs_reference_max_abs=r['parity']['batch1_vs_reference_max_abs'])

def cases(records):
    rows=[]
    for t in TASKS:
        for s in SEEDS:
            b=records[t,s,'mcmoe'];ids=b['ids'];labels=np.asarray(b['predictions']['label']);yr=rankdata(labels)
            ps={m:np.asarray(records[t,s,m]['predictions']['prediction']) for m in ALL}
            ranks={m:rankdata(v) for m,v in ps.items()}
            for i,video in enumerate(ids):
                rows.append(dict(task=t,seed=s,video_id=video,label=float(labels[i]),target_rank=float(yr[i]),
                    **{m+'_prediction':float(ps[m][i]) for m in ALL},
                    **{m+'_rank_error':float(abs(ranks[m][i]-yr[i])) for m in ALL},
                    eu_rank_error_reduction=float(abs(ranks['mcmoe'][i]-yr[i])-abs(ranks['mc_u'][i]-yr[i]))))
    groups={}
    for r in rows:groups.setdefault((r['task'],r['video_id']),[]).append(r)
    summaries=[]
    for (t,v),rs in groups.items():
        ds=[r['eu_rank_error_reduction'] for r in sorted(rs,key=lambda r:r['seed'])]
        summaries.append(dict(task=t,video_id=v,mean_rank_error_reduction=float(np.mean(ds)),seed_gains=ds))
    good=sorted(summaries,key=lambda r:(-r['mean_rank_error_reduction'],r['task'],r['video_id']))[:2]
    bad=sorted(summaries,key=lambda r:(r['mean_rank_error_reduction'],r['task'],r['video_id']))[:2]
    return rows,dict(selection='Two largest mean improvements and two largest mean degradations across ALL three seeds; descriptive, not representative prevalence or causal explanation.',
        success=good,failure=bad,semantic_review='Pending local video review. Numeric rank changes alone cannot establish action-level causes.')

def build(root):
    from run import validate_control,deps,validate_benchmark,bench_root
    root=Path(root);manifest=read(root/'manifest.json')
    if manifest['policy']!=POLICY:raise ValueError('Paper contract differs')
    records={};source_rows=[];training_rows=[]
    for t in TASKS:
        for s in SEEDS:
            for m in ALL:
                if m in METHODS:
                    out=completed(cell(root,'formal',t,s,m),True)
                    if out is None:raise ValueError('Cannot finalize paper tables: unfinished control '+str((t,s,m)))
                    r=validate_control(out,'formal',t,s,m,deps(root,'formal',t,s,m))
                else:
                    out=reference(root,'formal',t,s,m);r=read(out/'test_result.json')
                check_prediction(r);records[t,s,m]=r
                source_rows.append(dict(task=t,seed=s,method=m,path=str(out/'test_result.json'),sha256=sha(out/'test_result.json')))
                training_rows.append(dict(task=t,seed=s,method=m,epochs=r.get('epochs'),selected_epoch=r.get('selected_epoch'),
                    training_seconds=r.get('training_seconds'),wall_seconds=r.get('wall_seconds'),parameters=r.get('parameters'),
                    note='Extra frozen-parent branch training is not free; elapsed time may include shared-GPU interference.'))
            b=records[t,s,'mcmoe']
            for m in ALL:
                r=records[t,s,m]
                if r['ids']!=b['ids'] or r['predictions']['label']!=b['predictions']['label']:
                    raise ValueError('Unpaired evaluation order/labels '+str((t,s,m)))
    costs=[]
    for t in TASKS:
        for s in SEEDS:
            for parent in ('mcmoe','mc_ordered'):
                c=bench_root(root,'formal',True)/t/('seed%d'%s)/parent;out=completed(c,True);exclusive=out is not None
                if not out:out=completed(bench_root(root,'formal',False)/t/('seed%d'%s)/parent,True)
                if out is None:raise ValueError('Missing formal benchmark '+str((t,s,parent)))
                r=validate_benchmark(out,root,'formal',t,s,parent,exclusive);costs.append(audit_cost(r))
                source_rows.append(dict(task=t,seed=s,method='benchmark_'+parent,path=str(out/'benchmark.json'),sha256=sha(out/'benchmark.json')))
    overall,seeds,tasks,pairs=summarize(records);case_rows,queue=cases(records)
    folder=root/'paper';folder.mkdir(exist_ok=True)
    for name,rs in [('overall',overall),('per_seed',seeds),('per_score',tasks),('cost',costs),('training_cost',training_rows),('sources',source_rows),('cases_all',case_rows)]:csv_write(folder/(name+'.csv'),rs)
    save(folder/'design_evidence.json',pairs);save(folder/'case_queue.json',queue)
    save(folder/'paired_predictions.json',list(records.values()))
    open_items=['Finalize authorship, expert-reviewed video interpretation, bibliography and IEEE layout.',
        'Verify current strong-method feature/selection compatibility; published numbers are a separate literature table.',
        'Current official-val evaluation-best protocol is preserved, not an unseen test. Do not claim blind-test generalization.',
        'Ordered has not shown a stable added benefit on top of strong MCMoE; limit universal synergy claims.']
    if not all(c['exclusive'] for c in costs):open_items.append('Exclusive GPU timing remains incomplete; shared-GPU cost is diagnostic only.')
    if any(p['mean_gain']<=0 for p in pairs):open_items.append('At least one strong-backbone design comparator does not favor the proposed design; narrow the claim rather than hiding it.')
    cost_summary={p:dict(cells=21,observed_exclusive=sum(c['exclusive'] for c in costs if c['parent']==p),
        median_u_overhead_pct=float(np.median([c['deploy_u_overhead_pct'] for c in costs if c['parent']==p])),
        median_deployment_speedup_pct=float(np.median([c['deployment_speedup_pct'] for c in costs if c['parent']==p]))) for p in ('mcmoe','mc_ordered')}
    save(folder/'gaps.json',dict(engineering_complete=True,scientific_success_not_assumed=True,open_items=open_items,cost=cost_summary))
    text=['# Verified v39 experimental tables','',
        'SRCC x100. Seven FS1000 scores, three fixed seeds. Old training and evaluation-best selection unchanged.','',
        '| Method | Mean | Seed SD | vs MCMoE |','|---|---:|---:|---:|']
    text += ['| %s | %.3f | %.3f | %+.3f |'%(r['name'],r['mean'],r['seed_sd'],r['delta_vs_mcmoe']) for r in overall]
    text+=['','## Targeted design evidence','']
    for r in pairs:text+=['- %s minus %s: %+.4f; seed gains %s; wins/ties/losses %d/%d/%d. %s.'%(r['proposed'],r['control'],r['mean_gain'],r['seed_gains'],r['wins'],r['ties'],r['losses'],r['interpretation'])]
    text+=['','## Cost scope','', 'Cached feature forward only (batch1, 95 clips); raw-video feature extraction excluded. Parameters and extra 500+500 training epochs are not inference latency.','',json.dumps(cost_summary,indent=2),'','## Still open','']+['- '+v for v in open_items]
    (folder/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    for name in ('manuscript.tex','supplement.tex','references.bib','novelty_and_claims.md','submission_checklist.md','literature.json'):
        source=HERE/'paper'/name
        if source.exists():shutil.copyfile(source,folder/name)
    history=HERE/'paper/history'
    if history.exists():shutil.copytree(history,folder/'history',dirs_exist_ok=True)
    table=['% Automatically audited from predictions; SRCC x100.','\\begin{tabular}{lrrr}','\\hline','Method & Mean & SD & Gain \\\\','\\hline']
    table+=['%s & %.3f & %.3f & %+.3f \\\\'%(r['name'],r['mean'],r['seed_sd'],r['delta_vs_mcmoe']) for r in overall]
    table+=['\\hline','\\end{tabular}']
    (folder/'v39_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print('PAPER_TABLES_VERIFIED: 168 scores, 42 paired replay/timing records; see paper/gaps.json',flush=True)
    return dict(scores=168,cost_records=42,pairs=pairs,cost=cost_summary)

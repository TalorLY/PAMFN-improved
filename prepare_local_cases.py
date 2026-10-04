"""Local-only source-video contact sheets from audited v38 per-video results.

No videos/features uploaded. OpenCV and Pillow are needed only on this local
machine, not in the server training environment. No inferred action labels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--videos',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(Path(a.csv).open(encoding='utf-8-sig',newline='')));groups={}
    for r in rows:groups.setdefault((r['task'],r['video_id']),[]).append(r)
    values=[]
    for (task,vid),rs in groups.items():
        rs=sorted(rs,key=lambda r:int(r['seed']))
        if [int(r['seed']) for r in rs]!=[0,1,2]:raise ValueError('All three seeds required')
        b=[float(r['mcmoe_rank_error']) for r in rs];u=[float(r['mc_u_rank_error']) for r in rs]
        values.append(dict(task=task,video_id=vid,base_rank_errors=b,eu_rank_errors=u,
            seed_gains=[x-y for x,y in zip(b,u)],mean_gain=mean(x-y for x,y in zip(b,u))))
    selected=[dict(kind='success',**v) for v in sorted(values,key=lambda v:(-v['mean_gain'],v['task'],v['video_id']))[:2]]
    selected += [dict(kind='failure',**v) for v in sorted(values,key=lambda v:(v['mean_gain'],v['task'],v['video_id']))[:2]]
    wanted={r['video_id'] for r in selected};paths={v:[] for v in wanted}
    for v in Path(a.videos).rglob('*.mp4'):
        if v.stem in wanted:paths[v.stem].append(v)
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    sheets=[]
    for index,row in enumerate(selected):
        candidates=sorted(paths[row['video_id']],key=lambda p:str(p))
        if not candidates:row['visual_status']='source_video_not_found';continue
        src=candidates[0];cap=cv2.VideoCapture(str(src));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        if count<=0 or fps<=0:raise ValueError('Unreadable video '+str(src))
        canvas=Image.new('RGB',(1200,344),'#f8fafc');d=ImageDraw.Draw(canvas)
        d.text((14,10),'%s | %s | %s'%(row['kind'].upper(),row['task'],row['video_id']),fill='#15283e',font=font)
        d.text((14,40),'Rank-error reduction: mean %+.2f | seeds %s'%(row['mean_gain'],', '.join('%+.1f'%x for x in row['seed_gains'])),fill='#15283e',font=small)
        timestamps=[]
        for j,fraction in enumerate((.25,.5,.75)):
            frame=int((count-1)*fraction);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read()
            if not ok:raise ValueError('Failed frame '+str((src,frame)))
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);im=Image.fromarray(rgb);im.thumbnail((384,216))
            canvas.paste(im,(12+j*398+(384-im.width)//2,77+(216-im.height)//2))
            timestamps.append(frame/fps);d.text((14+j*398,300),'t = %.1fs (context only)'%(frame/fps),fill='#15283e',font=small)
        cap.release();name='case_%02d_%s.png'%(index+1,row['kind']);canvas.save(out/name);sheets.append(canvas)
        digest=hashlib.sha256()
        with src.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        row.update(source_video=str(src),source_sha256=digest.hexdigest(),source_candidates=[str(v) for v in candidates],
            duration_seconds=count/fps,frame_times_seconds=timestamps,image=name,visual_status='frames_extracted_context_only',
            semantic_review='Pending: frames are uniform context samples, NOT evidence of a localized error or modality causal mechanism.')
    if sheets:
        all_image=Image.new('RGB',(1200,344*len(sheets)),(255,255,255))
        for i,sheet in enumerate(sheets):all_image.paste(sheet,(0,344*i))
        all_image.save(out/'case_contact_sheet.png')
    result=dict(source_csv=str(Path(a.csv).resolve()),source_csv_sha256=hashlib.sha256(Path(a.csv).read_bytes()).hexdigest(),
        selection='Two greatest average improvements and two greatest degradations in MCMoE+E/U rank error over three seeds; deterministic task/id tie breaks; all seven scores considered.',
        selection_population=len(values),cases=selected)
    (out/'case_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

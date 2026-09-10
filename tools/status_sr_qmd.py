"""Read-only file progress. Does not infer process liveness from log timestamps."""
import argparse
from pathlib import Path
import re

from tools.summarize_sr_qmd import read_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='results/sr_qmd_ordered_v1_screen')
    args=p.parse_args()
    root=Path(args.root)
    if not (root/'manifest.json').exists():
        print('NOT_STARTED: manifest not created; check the pipeline/smoke launcher log')
        return
    manifest=read_json(root/'manifest.json')
    cells=[root/a/('seed%d'%s)/m for a in manifest['actions'] for s in manifest['seeds'] for m in manifest['methods']]
    done=[c for c in cells if (c/'complete.json').is_file()]
    print('Completed: %d/%d'%(len(done),len(cells)))
    for cell in cells:
        if cell in done:
            continue
        log=cell/'train.log'
        if log.is_file():
            epochs=re.findall(r'^Epoch\[(\d+)/(\d+)\].*$',log.read_text(encoding='utf-8',errors='replace'),re.M)
            latest='epoch %d/%s (display converted to 1-based)'%(int(epochs[-1][0])+1,epochs[-1][1]) if epochs else 'loading/startup'
            print('UNFINISHED %s: last recorded %s'%(cell.relative_to(root),latest))
    report=root/'report/completion.json'
    print('Completion report:',read_json(report) if report.exists() else 'not yet created')
    for suffix in ('_pilot_return.zip','_return.zip'):
        archive=Path(str(root)+suffix)
        print('Archive:',str(archive), 'READY' if archive.is_file() else 'not yet packaged')
    print('Use pgrep/nvidia-smi to check live processes; an unfinished log can also mean interruption.')


if __name__=='__main__':
    main()

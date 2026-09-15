"""Read-only v8 result import for the validation-only gradient fix1 release."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import shutil
import subprocess

from tools import run_qmd_dual as core
from tools.qmd_process import alive

V8_COMMITS = dict(modern='83aa9b67df5e79c2751b6bf9284012ed3be7bee3',
                  july28='de068118e1a97bc10806d7e985403dff6c39d0f9')
FIX1_GUARD_SHA256 = '524d3c43338179bceedce999236ef998ccd2c7f4baf0c3be7a9c4d3e2a625361'
ALLOWED_CHANGES = {'tools/qmd_gradient.py', 'tests/test_qmd_gradient.py'}
MODERN_CHANGES = {'tools/run_qmd_gradient.py', 'tools/run_qmd_gradient.sh',
                  'tools/reuse_qmd_gradient_v8.py', 'tests/test_qmd_gradient_tools.py',
                  'QMD_GRAD_V8_FIX1.md'}


def verify_source(old, new, repos):
    if old.get('commits') != V8_COMMITS:
        raise ValueError('reuse requires the exact original v8 commits')
    if any(old[k] != new[k] for k in ('inputs', 'environment')):
        raise ValueError('reuse data/environment changed')
    for protocol, repo in repos.items():
        changed = subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only',
                                          V8_COMMITS[protocol], 'HEAD'], text=True).splitlines()
        allowed = ALLOWED_CHANGES | (MODERN_CHANGES if protocol == 'modern' else set())
        if set(changed) - allowed:
            raise ValueError('reuse rejected: training code outside fix1 changed')
        guard = (repo/'tools/qmd_gradient.py').read_text(encoding='utf-8').encode()
        if hashlib.sha256(guard).hexdigest() != FIX1_GUARD_SHA256:
            raise ValueError('reuse requires the tested validation-only gradient patch')


@contextmanager
def source_lock(root):
    import fcntl  # Linux server only; never truncate or remove the old lock.
    with (root/'.runner.lock').open('r') as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        records = [root/'runner.json'] + list(root.rglob('*.process.json'))
        for path in records:
            if 'attempts' not in path.relative_to(root).parts and path.is_file():
                if alive(core.read(path).get('identity')):
                    raise RuntimeError('old v8 process still alive: ' + str(path))
        yield


def validate_completed(out, cell):
    from tools import run_qmd_gradient as run
    record = core.read(out/'complete.json')
    row = run.validate_cell(out, cell)
    if row != record['metrics']:
        raise ValueError('reused cell metrics changed: ' + str(out))
    config = dict(argv=run.cell_args(cell, '0')[5:], protocol=cell[0])
    if core.read(out/'run_config.json') != config:
        raise ValueError('reused cell arguments changed: ' + str(out))
    hashes = dict(record['files'], **{'model.pth':record['checkpoint_sha256']})
    if cell[0] == 'july28':
        hashes['model.pth.eval_rng.pt'] = record['eval_rng_sha256']
    for name, digest in hashes.items():
        if Path(name).name != name or core.sha(out/name) != digest:
            raise ValueError('reused cell artifact changed: ' + name)
    return record


def copy_unchanged(src, dst):
    """Recoverable/idempotent copy; no hard links and no existing file overwrite."""
    if dst.exists():
        if core.sha(src) != core.sha(dst):
            raise ValueError('reuse destination already differs: ' + str(dst))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(dst.name + '.reuse.tmp')
    shutil.copyfile(src, temporary)
    if core.sha(src) != core.sha(temporary):
        raise ValueError('reuse copy verification failed: ' + str(src))
    temporary.replace(dst)


def reuse_completed(origin, root, repos, source, jobs):
    from tools import run_qmd_gradient as run
    origin, root = Path(origin).resolve(), Path(root).resolve()
    if origin == root or origin in root.parents or root in origin.parents:
        raise ValueError('reuse requires separate, non-nested result roots')
    manifest_path = root/'reuse_manifest.json'
    if manifest_path.exists():
        previous = core.read(manifest_path)
        if previous['origin'] != str(origin) or previous['destination_source'] != source:
            raise ValueError('reuse source/root changed after import')
        print('REUSE: %d original v8 cells already imported.' % len(previous['cells']), flush=True)
        return
    with source_lock(origin):
        old = core.read(origin/'source.json')
        verify_source(old, source, repos)
        if core.read(origin/'manifest.json') != core.read(root/'manifest.json'):
            raise ValueError('reuse experiment matrix/recipe changed')
        selected = []
        for cell in jobs:
            out = core.cell_path(origin/'screen', cell)
            if not (out/'complete.json').is_file():
                continue  # Failed/unfinished training is never treated as complete.
            record = validate_completed(out, cell)
            if run.mode_for(cell) == 'observe':
                run.check_observe_reference(out, cell, root/'reference')
            files = [p for p in out.iterdir() if p.is_file()]
            selected.append((cell, out, record, files))
        needed = sum(p.stat().st_size for _,_,_,files in selected for p in files)
        if shutil.disk_usage(root).free < needed + 1024**3:
            raise RuntimeError('insufficient space for independent reused checkpoint copies')
        imported = []
        for cell, out, record, files in selected:
            dest = core.cell_path(root/'screen', cell)
            # complete.json last: a partially copied cell is never marked ready.
            for path in sorted(files, key=lambda p: (p.name == 'complete.json', p.name)):
                copy_unchanged(path, dest/path.name)
            validate_completed(dest, cell)
            imported.append(dict(cell=list(cell), original_complete_sha256=core.sha(out/'complete.json'),
                                 checkpoint_sha256=record['checkpoint_sha256']))
            print('REUSE VERIFIED ' + '/'.join(map(str, cell)), flush=True)
        # Preserve the original failure evidence in the final small return ZIP.
        for name in ('source.json','manifest.json','runner.json','launcher.log','screen/failure.json'):
            path = origin/name
            if path.is_file(): copy_unchanged(path, root/'original_v8'/name)
        for cell in jobs:
            out = core.cell_path(origin/'screen', cell)
            if (out/'complete.json').is_file(): continue
            for name in ('train.log','failure.json','train.process.json','model.pth.epochs.jsonl'):
                path = out/name
                if path.is_file(): copy_unchanged(path, root/'original_v8'/path.relative_to(origin))
        core.save(manifest_path, dict(origin=str(origin), source=old, destination_source=source,
                  cells=imported, reason='fix1 changes only the projection error check; successful v8 updates are identical'))
        print('REUSE COMPLETE: %d/%d formal cells; %d still need training. Old root unchanged.' %
              (len(imported),len(jobs),len(jobs)-len(imported)), flush=True)

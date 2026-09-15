"""PID identities for Linux server progress; no signalling or process mutation."""
import os
from pathlib import Path
import socket


def identity(pid):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return dict(host=socket.gethostname(), pid=pid, start_ticks=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    state=fields[0])
    except (OSError, IndexError):
        return None


def alive(record):
    if not record or record.get('host') != socket.gethostname():
        return None
    current = identity(record.get('pid', -1))
    if current is None:
        return False if os.name == 'posix' else None
    return (all(record.get(k) == current[k] for k in ('pid', 'start_ticks', 'boot_id'))
            and current['state'] not in ('Z', 'X'))

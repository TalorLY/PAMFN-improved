"""Import the byte-preserved v38 engine without touching installed packages."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
LEGACY = HERE / 'legacy'
if str(LEGACY) not in sys.path:
    sys.path.append(str(LEGACY))

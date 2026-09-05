"""Package the local release candidate; omit private interview preparation notes."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
reference = json.loads((root / 'evidence/reference_v0_1_1/manifest.json').read_text())
repeat = json.loads((root / 'runs/repeat_v0_1_1/manifest.json').read_text())
keys = ['semantic_result_sha256', 'calibration_sha256', 'dataset_hashes', 'source_sha256']
assert all(reference[key] == repeat[key] for key in keys), 'Reproducibility check failed'
excluded = {'runs', '__pycache__', '.git', '.venv', 'dist', 'build'}
private = {'INTERVIEW_PREP.md', 'PORTFOLIO_RECONCILIATION.md'}
files = sorted(p for p in root.rglob('*') if p.is_file()
               and not any(part in excluded or part.endswith('.egg-info') for part in p.relative_to(root).parts)
               and p.name not in private and p.suffix != '.pyc')
target = root.parent / 'synthetic-ts-benchmark-v0.1.1.zip'
with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in files:
        info = zipfile.ZipInfo('synthetic-ts-benchmark/' + path.relative_to(root).as_posix(),
                               date_time=(2026, 9, 5, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, path.read_bytes())
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
print(json.dumps({'archive': str(target), 'files': len(files),
                  'bytes': target.stat().st_size,
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  'reference_repeat_match': True}, indent=2))

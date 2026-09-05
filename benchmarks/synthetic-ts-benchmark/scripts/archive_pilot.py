"""Preserve the exact local pilot sources before a protocol correction."""
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
target=root/'evidence/reference/source_snapshot'
if target.exists():raise SystemExit('Pilot snapshot already exists; will not overwrite')
shutil.copytree(root/'tsbench',target/'tsbench',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(root/'configs/benchmark.json',target/'config.json')
shutil.copy2(root/'requirements.txt',target/'requirements.txt')
print(target)

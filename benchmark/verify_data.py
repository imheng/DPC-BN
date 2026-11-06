"""Verify exact experiment input bytes against the historical manifest."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
manifest=json.loads(Path(__file__).with_name('dataset_manifest.json').read_text())
fail=[]
for row in manifest:
 f=a.data/row['file']
 if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:fail.append(row['file'])
if fail:raise SystemExit('Missing or different files: '+', '.join(fail))
print(f'All {len(manifest)} historical input hashes match.')

"""Check portable core/evaluation assets without changing data or downloading models."""
import importlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    failures = []
    if struct.calcsize('P') != 8:
        failures.append('64-bit Python required')
    for name in ('numpy', 'pandas', 'torch', 'transformers', 'yaml'):
        try:
            importlib.import_module(name)
        except Exception as error:
            failures.append(f'{name}: {error}')
    required = ['SLclean.sqlite3', 'pose-animator-dist/dataset_player.html',
                'recognition_audit/wlasl_stgcn.zip', 'recognition_audit/wlasl_metadata.zip',
                'recognition_audit/slgcn_trial_20261002/wlasl_slgcn.zip']
    required += ['models/xlsr53-english/'+name for name in
                 ('config.json', 'model.safetensors', 'preprocessor_config.json',
                  'special_tokens_map.json', 'vocab.json')]
    for name in required:
        if not (ROOT/name).is_file():
            failures.append(f'Missing: {name}')
    cases = list((ROOT/'recognition_audit/slgcn200_20261002/cases').glob('*.json'))
    if len(cases) != 200:
        failures.append(f'Experiment 1 expects 200 cases; found {len(cases)}')
    directory = ROOT/'evaluation/model-results/oracle200-20261011'
    try:
        manifest = json.loads((directory/'capture_manifest.json').read_text(encoding='utf-8'))
        if len(manifest) != 200:
            failures.append('Experiment 2/3 manifest must contain 200 clips')
        for item in manifest:
            if not (directory/'inference'/f"{item['id']}.npz").is_file():
                failures.append(f"Missing inference input: {item['id']}")
    except (OSError, ValueError) as error:
        failures.append(f'Experiment 2/3 manifest: {error}')
    for error in failures:
        print('FAIL:', error)
    if failures:
        return 1
    print('Core and Experiment 1-3 assets/dependencies OK. This check does not test microphone permissions or model inference.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

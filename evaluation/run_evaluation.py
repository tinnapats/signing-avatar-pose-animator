"""Re-run pretrained recognition on archived experiment inputs, never overwrite archives.

Experiment 1: pre-render payload. Experiment 2: image-only RTMW input.
Experiment 3: renderer-assisted input (NOT image-only).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'recognition_audit'
sys.path.insert(0, str(AUDIT))


def metrics(rows):
    labels = sorted({r['label'] for r in rows})
    scores = []
    for label in labels:
        tp = sum(r['label'] == label == r['prediction'] for r in rows)
        fp = sum(r['label'] != label == r['prediction'] for r in rows)
        fn = sum(r['label'] == label != r['prediction'] for r in rows)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        scores.append((p, r, 2*p*r/(p+r) if p+r else 0))
    return dict(count=len(rows), correct=sum(r['label'] == r['prediction'] for r in rows),
                accuracy=sum(r['label'] == r['prediction'] for r in rows)/len(rows),
                **dict(zip(('precision', 'recall', 'f1'),
                           [sum(s[i] for s in scores)/len(scores) for i in range(3)])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--model', choices=('stgcn', 'slgcn'), default='stgcn')
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.limit <= 200:
        parser.error('--limit must be between 1 and 200')
    import numpy as np
    import torch
    import run_audit as audit
    model, normalizer, vocab = audit.load_model()
    if args.model == 'slgcn':
        import compare_slgcn_best
        model, _ = compare_slgcn_best.load_slgcn()
    rows = []
    if args.experiment == 1:
        cases = sorted((AUDIT/'slgcn200_20261002/cases').glob('*.json'))
        if len(cases) != 200:
            raise RuntimeError('Expected the complete local 200-clip archive')
        def inputs():
            for path in cases[:args.limit]:
                case = json.loads(path.read_text(encoding='utf-8'))
                arr = audit.frame_array([case['payload']['frames'][i] for i in case['frameIndices']])
                tensor = torch.from_numpy(arr[:, audit.PICK].transpose(2, 0, 1).copy())
                yield case['label'], normalizer({'frames': tensor})['frames']
    else:
        directory = ROOT/'evaluation/model-results/oracle200-20261011'
        manifest = json.loads((directory/'capture_manifest.json').read_text(encoding='utf-8'))
        key = 'pixels_input' if args.experiment == 2 else 'both_oracle_input'
        def inputs():
            for case in manifest[:args.limit]:
                # These arrays already contain the original shoulder normalization.
                # Do not normalize a second time or use renderer data for Exp 2.
                with np.load(directory/'inference'/f"{case['id']}.npz", allow_pickle=False) as data:
                    yield case['label'], torch.from_numpy(data[key].copy())
    for label, tensor in inputs():
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Non-finite input: {label}')
        with torch.inference_mode():
            p = model(tensor.unsqueeze(0))[0].softmax(0)
        top = p.topk(5).indices.tolist()
        rows.append(dict(label=label, prediction=vocab[top[0]], top5=[vocab[i] for i in top]))
        print(f"{len(rows)}/{args.limit}: {label} -> {rows[-1]['prediction']}", flush=True)
    result = dict(experiment=args.experiment, model=args.model, metrics=metrics(rows),
                  scope='Fresh inference on archived inputs; not fresh image extraction or an independent test. '
                        'Experiment 3 uses renderer assistance. Scores are recognition, not visual similarity.',
                  predictions=rows)
    out = ROOT/'evaluation/runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    (out/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result['metrics'], indent=2))
    print(f'Report: {out / "results.json"}')


if __name__ == '__main__':
    main()

"""Recognition-only rerun on saved captures; never edits production or old reports."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from run_audit import HERE, PICK, load_model, frame_array
from step1_evaluate import hashes


def main():
    before = hashes()
    model, normalize, labels = load_model()
    predictions, summaries, evidence = [], [], {}
    for folder in ['step1b', 'ik_trial']:
        out = HERE / folder
        assert _local_json_loads((out/'production_before.json').read_text()) == before
        inputs = out/'inputs.json' if folder == 'ik_trial' else HERE/'inputs.json'
        captures = out/'rendered.json'
        for path in [inputs, captures]:
            evidence[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases = {c['phrase']: c for c in _local_json_loads(inputs.read_text())}
        records = _local_json_loads(captures.read_text())
        seen = set()
        for record in records:
            variant, name = record['variant'], record['phrase']
            assert (variant, name) not in seen
            seen.add((variant, name))
            case = cases[name]
            payload = case['ikPayload' if folder == 'ik_trial' and variant == 'IK' else 'payload']
            data = frame_array([payload['frames'][i] for i in case['frameIndices']])
            assert len(data) == len(record['samples'])
            for t, sample in enumerate(record['samples']):
                for side, offset in [('left', 33), ('right', 54)]:
                    data[t, offset:offset+21] = 0
                    for index, point in sample['hands'][side].items():
                        data[t, offset+int(index)] = point
            tensor = torch.from_numpy(data[:, PICK, :].transpose(2, 0, 1).copy())
            tensor = normalize({'frames': tensor})['frames'].unsqueeze(0)
            assert torch.isfinite(tensor).all()
            with torch.inference_mode():
                logits = model(tensor)[0]
            assert torch.isfinite(logits).all()
            top = [labels[i] for i in logits.topk(5).indices.tolist()]
            predictions.append(dict(group=folder, variant=variant, label=name, prediction=top[0], top5=top))
        for variant in sorted({r['variant'] for r in records}):
            rows = [r for r in predictions if r['group'] == folder and r['variant'] == variant]
            assert len(rows) == len(cases) == 30
            scores = []
            for name in cases:
                tp = sum(r['label'] == name and r['prediction'] == name for r in rows)
                fp = sum(r['label'] != name and r['prediction'] == name for r in rows)
                fn = sum(r['label'] == name and r['prediction'] != name for r in rows)
                p = tp/(tp+fp) if tp+fp else 0
                r = tp/(tp+fn) if tp+fn else 0
                scores.append([p, r, 2*p*r/(p+r) if p+r else 0])
            p, r, f = np.mean(scores, axis=0)*100
            summary = dict(group=folder, variant=variant, correct=sum(r['prediction']==r['label'] for r in rows), macro_precision=float(p), macro_recall=float(r), macro_f1=float(f))
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    assert hashes() == before
    baseline = {r['label']: r['prediction'] for r in predictions if r['group']=='ik_trial' and r['variant']=='BASELINE'}
    control = {r['label']: r['prediction'] for r in predictions if r['group']=='step1b' and r['variant']=='S1A1'}
    assert baseline == control, 'Baseline predictions disagree'
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), torch=torch.__version__, production_unchanged=True, evidence=evidence, summaries=summaries, predictions=predictions)
    (HERE/'recognition_comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Recognition comparison: 30 saved clips', '', '| Group | Variant | Correct / 30 | Macro Precision | Macro Recall | Macro F1 |', '|---|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['group']} | {s['variant']} | {s['correct']} | {s['macro_precision']:.2f}% | {s['macro_recall']:.2f}% | {s['macro_f1']:.2f}% |")
    lines += ['', 'Hands are captured renderer joint positions; body is from the matching payload (IK uses modified elbows). Not recognition of final avatar pixels or human comprehension.', '', 'Macro scores average over the 30 target labels with zero division set to zero, matching previous audits. One clip per label; unknown training overlap; not held-out evidence. All methods use identical clips and model. No production files changed.', '', 'Geometry and elbow-continuity results remain in step1b and ik_trial reports; improved recognition alone does not establish visual or biomechanical quality. Earlier reports saying PyTorch is blocked describe the previous run; this rerun succeeded.', '']
    (HERE/'RECOGNITION_COMPARISON.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

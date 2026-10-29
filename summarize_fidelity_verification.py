"""Summarize the explicitly selected source or this checkout; preserve old archives."""
import os
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE = Path(os.environ.get('FIDELITY_SOURCE', str(Path(__file__).resolve().parent))).resolve()
OUT = Path(os.environ.get('FIDELITY_OUTPUT', str(SOURCE / 'fidelity_verification' / 'current'))).resolve()

def xy(k):
    return np.array([k['position']['x'], k['position']['y']], dtype=float)

def score(errors, threshold):
    return float(np.mean(np.asarray(errors) <= threshold) * 100)

def main():
    # Sanity checks: identity must score 100; missing outputs must score zero.
    assert score([0, 0], .05) == 100
    assert score([float('inf')], .05) == 0
    assert score([.04, .06], .05) == 50
    cases = json.loads((SOURCE/'fidelity_results/inputs.json').read_text())
    rendered = json.loads((OUT/'rendered.json').read_text())
    rows, pooled = [], {k: [] for k in ['position', 'shape', 'backend_position', 'backend_shape', 'pixels']}
    for case, result in zip(cases, rendered, strict=True):
        assert case['phrase'] == result['phrase']
        errors = {k: [] for k in pooled}
        for ref, index, sample in zip(case['reference'], case['frameIndices'], result['samples'], strict=True):
            assert sample['viewMatches']
            pose = {k['part']: k for k in ref['pose']['keypoints']}
            if min(pose[s]['score'] for s in ['leftShoulder', 'rightShoulder']) <= 0:
                continue
            shoulder = np.linalg.norm(xy(pose['leftShoulder'])-xy(pose['rightShoulder']))
            if shoulder < 1e-6:
                continue
            for side, hand in ref.get('hands', {}).items():
                if not hand.get('observed'):
                    continue
                keys = hand['keypoints']
                palm = np.linalg.norm(xy(keys[9])-xy(keys[0]))
                if palm < 1e-6:
                    continue
                predicted = sample['hands'][side]
                backend = case['payload']['frames'][index]['hands'][side]['keypoints']
                for i, key in enumerate(keys):
                    if key.get('score', 0) <= 0:
                        continue
                    q = predicted.get(str(i))
                    error = np.linalg.norm(np.asarray(q)-xy(key)) if q is not None else float('inf')
                    errors['position'].append(error/shoulder)
                    errors['backend_position'].append(np.linalg.norm(xy(backend[i])-xy(key))/shoulder)
                    if q is not None:
                        errors['pixels'].append(error*sample['zoom'])
                    if i:
                        shape_error = np.linalg.norm((np.asarray(q)-np.asarray(predicted['0']))-(xy(key)-xy(keys[0]))) if q is not None and '0' in predicted else float('inf')
                        errors['shape'].append(shape_error/palm)
                        errors['backend_shape'].append(np.linalg.norm((xy(backend[i])-xy(backend[0]))-(xy(key)-xy(keys[0])))/palm)
        def summarize(name, e):
            return dict(phrase=name, points=len(e['position']),
                        coverage=100*len(e['pixels'])/len(e['position']),
                        backend_position_pck5=score(e['backend_position'], .05),
                        render_position_pck5=score(e['position'], .05),
                        render_position_pck10=score(e['position'], .10),
                        render_position_pck20=score(e['position'], .20),
                        backend_shape_pck20=score(e['backend_shape'], .20),
                        render_shape_pck10=score(e['shape'], .10),
                        render_shape_pck20=score(e['shape'], .20),
                        render_shape_pck30=score(e['shape'], .30),
                        rmse_canvas_units=float(np.sqrt(np.mean(np.square(e['pixels'])))))
        row = summarize(case['phrase'], errors)
        row['reference_frames'] = len(result['samples'])
        row['distinct_canvas_checksums'] = len({s['checksum'] for s in result['samples']})
        rows.append(row)
        for key in pooled:
            pooled[key].extend(errors[key])
    rows.append(summarize('Pooled', pooled))
    provenance = dict(source=str(SOURCE), files={str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [SOURCE/'export_pose_animator_sequence.py', SOURCE/'pose-animator/dataset_player.js', SOURCE/'pose-animator/illustrationGen/illustration.js']},
        inputs_sha256=hashlib.sha256((SOURCE/'fidelity_results/inputs.json').read_bytes()).hexdigest(),
        canvas_sizes=sorted({(s['width'], s['height']) for r in rendered for s in r['samples']}))
    (OUT/'metrics.json').write_text(json.dumps(dict(provenance=provenance, results=rows), indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()

"""Frozen three-clip body-source ablation; no production writes."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys, json, hashlib, importlib.util
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
A = Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
B = A / 'pixels30_20261007/rtm_trial_20261008'
D = ROOT / 'tracking3/rtmw_anchor_trial'
O = D / 'stgcn_body_hybrid'
O.mkdir(exist_ok=True)
assert not (O / 'protocol.json').exists(), 'Preserve previous run'
sys.path.insert(0, str(A))
import run_audit as audit
spec = importlib.util.spec_from_file_location('rtm_eval', ROOT.parent / 'rtm-eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = _local_json_loads((ROOT / 'tracking3/results.json').read_text())
z = np.load(D / 'predictions.npz')
previous = _local_json_loads((D / 'stgcn/results.json').read_text())
sources = ['raw', 'rtmw_anchor', 'rtmw_temporal']
protocol = {
    'status': 'registered', 'clips': ['abdomen', 'hello', 'love'],
    'body_sources': ['SDPose', 'RTMW'], 'hand_sources': sources,
    'checkpoint_sha256': sha(A / 'wlasl_stgcn.zip'),
    'metadata_sha256': sha(A / 'wlasl_metadata.zip'),
    'hand_predictions_sha256': sha(D / 'predictions.npz'),
    'converter_sha256': sha(ROOT.parent / 'rtm-eval.py'),
    'audit_sha256': sha(A / 'run_audit.py'), 'script_sha256': sha(Path(__file__)),
    'cache_sha256': {c: sha(B / (c + '.npz')) for c in ['000', '001', '002']},
    'python': sys.version, 'torch': torch.__version__,
    'rules': 'Same 133-to-75-to-27 mapping, confidence .30, missing zeros, clip shoulder normalization, original 30FPS captured times; no resampling or training.',
    'intervention': 'Replace body coordinates AND confidence indices 0:91 with RTMW; preserve SDPose hand blocks and confidence exactly. Normalization recomputes from selected body.',
    'macro': 'Three true labels only, zero_division=0', 'production_edited': False,
}
(O / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
model, norm, vocab = audit.load_model()
assert len(vocab) == 2000
pred, inputs, logits = {}, {}, {}
old_inputs = np.load(D / 'stgcn/model_inputs.npz')
old_logits = np.load(D / 'stgcn/logits.npz')
for cid, label in [('000', 'abdomen'), ('001', 'hello'), ('002', 'love')]:
    ii = [i for i, r in enumerate(rows) if r['id'] == cid]
    cache = np.load(B / (cid + '.npz'))
    assert [rows[i]['frame'] for i in ii] == list(range(len(ii)))
    assert np.allclose(cache['times'], [rows[i]['time'] for i in ii])
    variants = {}
    for source in sources:
        k, s = z[source][ii].copy(), z[source + '_scores'][ii].copy()
        variants['sd_body_' + source] = (k.copy(), s.copy())
        k[:, :91] = cache['rtmw_det'][:, :91]
        s[:, :91] = cache['rtmw_det_scores'][:, :91]
        assert np.array_equal(k[:, 91:], z[source][ii, 91:])
        assert np.array_equal(s[:, 91:], z[source + '_scores'][ii, 91:])
        variants['rtmw_body_' + source] = (k, s)
    variants['rtmw_all_control'] = (cache['rtmw_det'], cache['rtmw_det_scores'])
    for variant, (k, s) in variants.items():
        a = ev.convert(k, s)
        t = norm({'frames': torch.from_numpy(a[:, audit.PICK].transpose(2, 0, 1).copy())})['frames']
        assert torch.isfinite(t).all()
        with torch.inference_mode():
            out = model(t.unsqueeze(0))[0]
            p = out.softmax(0)
        target = vocab.index(label)
        top = p.topk(5)
        rr = {'label': label, 'prediction': vocab[int(top.indices[0])],
              'top5': [vocab[int(j)] for j in top.indices],
              'target_rank': torch.argsort(p, descending=True).tolist().index(target) + 1,
              'target_probability': float(p[target]), 'frames': len(ii)}
        if variant.startswith('sd_body_') or variant == 'rtmw_all_control':
            key = variant.removeprefix('sd_body_') if variant != 'rtmw_all_control' else 'rtmw_control'
            assert np.array_equal(t.numpy(), old_inputs[f'{cid}_{key}'])
            assert np.allclose(out.numpy(), old_logits[f'{cid}_{key}'], rtol=1e-5, atol=1e-5)
            old = (previous['predictions'][key] if key != 'rtmw_control' else previous['controls'])
            assert rr['top5'] == next(r for r in old if r['label'] == label)['top5']
        pred.setdefault(variant, []).append(rr)
        inputs[f'{cid}_{variant}'] = t.numpy()
        logits[f'{cid}_{variant}'] = out.numpy()
        print(label, variant, rr['prediction'], rr['target_rank'], flush=True)
result = {'summary': {v: ev.metrics(rr) for v, rr in pred.items()},
          'predictions': pred, 'prior_controls_reproduced': True}
np.savez_compressed(O / 'model_inputs.npz', **inputs)
np.savez_compressed(O / 'logits.npz', **logits)
(O / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
protocol['status'] = 'completed'
protocol['prior_controls_reproduced'] = True
(O / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
lines = ['# ST-GCN: ลำตัว RTMW + มือ SDPose', '',
         'ทดลองจริง 3 คลิป 212 เฟรม: abdomen, hello, love ไม่แก้ระบบหลัก ไม่ฝึกโมเดลใหม่ ทำนายอิสระ 2,000 คลาส',
         'เปลี่ยนลำตัวพร้อม confidence เป็น RTMW โดยเก็บมือ SDPose เดิมทุกจุด การปรับศูนย์และสเกลใช้สูตรเดิมแต่คำนวณจากไหล่ของลำตัวที่เลือก จึงไม่ได้แยกผลของลำตัวออกจาก normalization',
         'คงเวลาเดิม เกณฑ์ confidence 0.30 และวิธีจัดการข้อมูลหายเดิม ตรวจพิกัด normalized และ logits ชุดควบคุมตรงกับรอบก่อน', '',
         '| วิธี | ถูก/3 | Accuracy % | Macro-P % | Macro-R % | Macro-F1 % | Top5 % |',
         '|---|---:|---:|---:|---:|---:|---:|']
for v, m in result['summary'].items():
    lines.append(f"| {v} | {m['correct']}/3 | {m['accuracy']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['top5']:.2f} |")
lines += ['', '| คำจริง | วิธี | คำทาย | อันดับคำจริง | Probability คำจริง |', '|---|---|---|---:|---:|']
for v, rr in pred.items():
    for r in rr:
        lines.append(f"| {r['label']} | {v} | {r['prediction']} | {r['target_rank']} | {r['target_probability']:.6e} |")
lines += ['', 'Macro เฉลี่ย 3 true labels เท่านั้น ไม่ใช่ macro ทั้ง 2,000 คลาส หนึ่งคลิปต่อคำและเป็นคลิปเดิมที่ใช้พัฒนา จึงห้ามสรุปความแม่นยำภาพรวมจากชุดนี้',
          'คะแนนรู้จำไม่ใช่เปอร์เซ็นต์ความเหมือนหรือความถูกต้องทาง ASL โมเดลรับ 7 จุดลำตัวและมือข้างละ 10 จุด ไม่ได้อ่านมือครบ 21 จุด']
(O / 'REPORT_TH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(result['summary']), flush=True)

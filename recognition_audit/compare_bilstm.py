"""Isolated released-model comparison; baseline data/code remain untouched."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import hashlib
import inspect
import io
import json
from pathlib import Path
import zipfile
import torch
import yaml
import run_audit as baseline
from step1_evaluate import hashes

HERE = Path(__file__).resolve().parent
OUT = HERE / 'model_comparison'
DATA = HERE / 'recognition_100'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_model():
    torch.set_num_threads(4)
    with zipfile.ZipFile(OUT/'wlasl_lstm.zip') as archive:
        config = yaml.safe_load(archive.read('wlasl/lstm/lstm.yaml'))
        safe = [(type('UnusedCheckpointCallback', (), {}), 'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'),
                (type('UnusedStoppingCallback', (), {}), 'pytorch_lightning.callbacks.early_stopping.EarlyStopping')]
        with torch.serialization.safe_globals(safe):
            ckpt = torch.load(io.BytesIO(archive.read('wlasl/lstm/epoch=109-step=49059.ckpt')), map_location='cpu', weights_only=True)
    namespace = {}
    for file in ['upstream_utils.py', 'upstream_pose_flattener.py', 'upstream_rnn.py']:
        source = (OUT/file).read_text().replace('from .utils import AttentionBlock', '')
        exec(compile(source, str(OUT/file), 'exec'), namespace)
    encoder = namespace['PoseFlattener'](in_channels=2, **config['model']['encoder']['params'])
    decoder = namespace['RNNClassifier'](n_features=54, num_class=2000, **config['model']['decoder']['params'])
    encoder.load_state_dict({k.removeprefix('model.encoder.'): v for k,v in ckpt['state_dict'].items() if k.startswith('model.encoder.')}, strict=True)
    decoder.load_state_dict({k.removeprefix('model.decoder.'): v for k,v in ckpt['state_dict'].items() if k.startswith('model.decoder.')}, strict=True)
    with zipfile.ZipFile(HERE/'wlasl_metadata.zip') as archive:
        labels = sorted(v['gloss'] for v in _local_json_loads(archive.read('splits/asl2000.json')))
    transforms = baseline.module('comparison_transforms', 'upstream_pose_transforms.py')
    normalize = transforms.CenterAndScaleNormalize(reference_points_preset='shoulder_mediapipe_holistic_minimal_27')
    return torch.nn.Sequential(encoder, decoder).eval(), normalize, labels

def main():
    before = hashes()
    files = [DATA/'inputs.json', DATA/'rendered.json', DATA/'results.json', HERE/'run_audit.py', OUT/'wlasl_lstm.zip']
    evidence = {str(p): sha(p) for p in files}
    protocol = dict(model='OpenHands released WLASL2000 BiLSTM attention', data='Same frozen 100 clips, seven conditions, batch size one, no fine tuning or sample filtering',
        limitation='Upstream RNN uses batch_first=False but receives N,T,F from PoseFlattener. Preserved exactly, not repaired without retraining/verification. Thus results are diagnostic, not a clean architectural ranking.', hashes=evidence)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    # Reuse the exact baseline experiment, changing only loader, output path,
    # checkpoint attribution and scope description. No baseline file is edited.
    source = inspect.getsource(baseline.run)
    source = source.replace("(audit_dir/'results.json')", "(COMPARISON_OUT/'bilstm_results.json')")
    source = source.replace("(HERE/'wlasl_stgcn.zip')", "(COMPARISON_OUT/'wlasl_lstm.zip')")
    source = source.replace('Pilot, one source clip per class; unknown train overlap.', 'Same frozen 100 released-test clips; upstream RNN axis caveat applies.')
    namespace = dict(vars(baseline), load_model=load_model, COMPARISON_OUT=OUT)
    exec(compile(source, 'baseline_run_output_redirect_only', 'exec'), namespace)
    namespace['run'](DATA)
    assert before == hashes(), 'Production changed during evaluation'
    assert all(sha(Path(p)) == h for p,h in evidence.items()), 'Baseline evidence changed'
    old = _local_json_loads((DATA/'results.json').read_text())
    new = _local_json_loads((OUT/'bilstm_results.json').read_text())
    assert len(new['predictions']) == 700
    assert old['labels_evaluated'] == new['labels_evaluated']
    lines = ['# Same 100 clips: ST-GCN versus released BiLSTM', '',
        'Diagnostic only: preserved upstream BiLSTM axis behavior; not evidence of human sign comprehension.', '',
        '| Condition | ST-GCN accuracy | BiLSTM accuracy | ST-GCN Macro-F1 | BiLSTM Macro-F1 |', '|---|---:|---:|---:|---:|']
    for a in old['summaries']:
        b = next(s for s in new['summaries'] if s['variant'] == a['variant'])
        lines.append(f"| {a['variant']} | {a['accuracy']:.2f}% | {b['accuracy']:.2f}% | {a['macro_f1']:.2f}% | {b['macro_f1']:.2f}% |")
    lines += ['', 'Macro averages over the same 100 target labels, zero division=0. Out-of-set predictions remain errors.',
        'Both models omit some finger joints. Rendered condition uses hand strokes plus payload body, not rendered-image recognition.',
        'No production, SQLite, baseline result, sample list or extraction changes.',
        'Official sources: https://github.com/AI4Bharat/OpenHands ; https://openhands.ai4bharat.org/en/latest/instructions/models.html',
        'Checkpoint/source hashes and per-clip predictions are retained alongside this report.']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT/'evidence.json').write_text(json.dumps({p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'evidence.json'}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()

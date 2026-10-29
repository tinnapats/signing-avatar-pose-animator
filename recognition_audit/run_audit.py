"""Isolated OpenHands ST-GCN evaluation; does not modify production.

Uses original upstream architecture and normalization. The sole in-memory
adaptations replace OmegaConf conversion with a plain config dictionary and
the relative graph import. Checkpoint tensors are loaded strictly. No training.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import random
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import export_pose_animator_sequence as pipeline

BODY = [(0, 'nose'), (2, 'leftEye'), (5, 'rightEye'), (11, 'leftShoulder'),
        (12, 'rightShoulder'), (13, 'leftElbow'), (14, 'rightElbow')]
PICK = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46,
        49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]

def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def load_model():
    torch.set_num_threads(4)
    archive = zipfile.ZipFile(HERE / 'wlasl_stgcn.zip')
    config = yaml.safe_load(archive.read('wlasl/st_gcn/config.yaml'))
    # These old callback classes occur as dictionary keys only. Inert stand-ins
    # let weights_only=True read tensors without importing/executing callbacks.
    safe = [(type('UnusedCheckpointCallback', (), {}), 'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'),
            (type('UnusedStoppingCallback', (), {}), 'pytorch_lightning.callbacks.early_stopping.EarlyStopping')]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(io.BytesIO(archive.read('wlasl/st_gcn/epoch=212-step=95210.ckpt')),
                                map_location='cpu', weights_only=True)
    graph = module('audit_graph', 'upstream_graph_utils.py')
    source = (HERE / 'upstream_st_gcn.py').read_text()
    source = source.replace('from omegaconf import OmegaConf', '')
    source = source.replace('from .graph_utils import GraphWithPartition', '')
    source = source.replace('graph_args = OmegaConf.to_container(graph_args)', 'graph_args = dict(graph_args)')
    namespace = {'GraphWithPartition': graph.GraphWithPartition}
    exec(compile(source, 'upstream_st_gcn_adapted', 'exec'), namespace)
    encoder = namespace['STGCN'](in_channels=2, **config['model']['encoder']['params'])
    sd = checkpoint['state_dict']
    assert sd['model.decoder.classifier.weight'].shape == (2000, 256)
    encoder.load_state_dict({k.removeprefix('model.encoder.'): v for k, v in sd.items() if k.startswith('model.encoder.')}, strict=True)
    decoder = torch.nn.Linear(256, 2000)
    decoder.load_state_dict({k.removeprefix('model.decoder.classifier.'): v for k, v in sd.items() if k.startswith('model.decoder.classifier.')}, strict=True)
    network = torch.nn.Sequential(encoder, decoder).eval()
    metadata = zipfile.ZipFile(HERE/'wlasl_metadata.zip')
    labels = sorted(v['gloss'] for v in json.loads(metadata.read('splits/asl2000.json')))
    transforms = module('audit_transforms', 'upstream_pose_transforms.py')
    normalizer = transforms.CenterAndScaleNormalize(reference_points_preset='shoulder_mediapipe_holistic_minimal_27')
    return network, normalizer, labels

def frame_array(frames):
    data = np.zeros((len(frames), 75, 2), dtype=np.float32)
    for t, frame in enumerate(frames):
        pose = {k['part']: k for k in frame['pose']['keypoints']}
        for index, name in BODY:
            k = pose[name]
            if k.get('score', 0) > 0:
                data[t,index] = [k['position']['x'], k['position']['y']]
        for side, offset in [('left', 33), ('right', 54)]:
            hand = frame.get('hands', {}).get(side, {})
            for i, k in enumerate(hand.get('keypoints', [])):
                if k.get('score', 0) > 0 and hand.get('score', 0) > 0:
                    data[t,offset+i] = [k['position']['x'], k['position']['y']]
    return data

def csv_array(df):
    frame_ids = sorted(df.frame.unique())
    index = {v:i for i,v in enumerate(frame_ids)}
    result = np.zeros((len(frame_ids),75,2),dtype=np.float32)
    offsets = {'pose':0,'left_hand':33,'right_hand':54}
    for row in df[df.part.isin(offsets)].itertuples():
        result[index[row.frame],offsets[row.part]+int(row.landmark_id)] = [row.x,row.y]
    return result

def prepare(count):
    db = ROOT/'SLclean.sqlite3'
    clips = pipeline.discover_clips(db)
    available = sorted(Path(name).stem for name in clips)
    required = ['abdomen','hello','love','test','yes','please','thank you','no']
    rng = random.Random(20260926)
    selected = required + rng.sample([x for x in available if x not in required], max(0,count-len(required)))
    cases = []
    for label in selected:
        paths = pipeline.resolve_selected_paths(clips,db,[label+'.csv'],'')
        native = csv_array(pipeline.read_clip_csv(paths[0]))
        payload = pipeline.build_payload(db,[label+'.csv'],'',30,513,513,0,0,2,1.2,2)
        source30 = csv_array(pipeline.concat_clips(paths,513,513,0,target_fps=30))
        indexes = list(range(0,len(payload['frames']),2))
        assert len(indexes)==len(source30)
        cases.append(dict(phrase=label,payload=payload,frameIndices=indexes,native=native.tolist(),source30=source30.tolist()))
        print('prepared',label,len(native),flush=True)
    (HERE/'inputs.json').write_text(json.dumps(cases),encoding='utf-8')

def run(data_dir=None):
    audit_dir=Path(data_dir) if data_dir is not None else HERE
    network, normalize, labels = load_model()
    cases = json.loads((audit_dir/'inputs.json').read_text())
    rendered = {r['phrase']:r['samples'] for r in json.loads((audit_dir/'rendered.json').read_text())} if (audit_dir/'rendered.json').exists() else {}
    predictions = []
    def predict(data, label, variant):
        tensor = torch.from_numpy(np.asarray(data,dtype=np.float32)[:,PICK,:].transpose(2,0,1).copy())
        tensor = normalize({'frames':tensor})['frames'].unsqueeze(0)
        assert torch.isfinite(tensor).all()
        start=time.perf_counter()
        with torch.inference_mode():
            logits = network(tensor)[0]
            probabilities = logits.softmax(0)
        top = probabilities.topk(5)
        return dict(label=label,variant=variant,prediction=labels[int(top.indices[0])],
                    top5=[labels[int(i)] for i in top.indices],target_probability=float(probabilities[labels.index(label)]),
                    runtime_ms=(time.perf_counter()-start)*1000)
    # Exclude warmup from timings.
    predict(np.asarray(cases[0]['native']),cases[0]['phrase'],'warmup')
    for case in cases:
        label=case['phrase']; native=np.asarray(case['native'],dtype=np.float32)
        processed=frame_array([case['payload']['frames'][i] for i in case['frameIndices']])
        variants={'source_native':native,'source_30fps':np.asarray(case['source30'],dtype=np.float32),'processed_30fps':processed}
        frozen=native.copy()
        for offset in [33,54]:
            valid=np.where(np.any(native[:,offset:offset+21]!=0,axis=(1,2)))[0]
            if len(valid):
                f=int(valid[0]); relative=native[f,offset:offset+21]-native[f,offset]
                frozen[valid,offset:offset+21]=native[valid,offset:offset+1]+relative
        variants['freeze_fingers']=frozen
        variants['shuffle_frames']=native[np.random.default_rng(20260926).permutation(len(native))]
        blind=native.copy()
        # Deliberately corrupt joints NOT consumed by the released checkpoint.
        unused=[i for i in range(33,75) if i not in PICK]
        blind[:,unused]+=1.0
        assert np.array_equal(blind[:,PICK],native[:,PICK])
        variants['corrupt_unselected_joints']=blind
        if label in rendered:
            samples=rendered[label]; assert len(samples)==len(processed)
            hybrid=processed.copy()
            for t,sample in enumerate(samples):
                for side,offset in [('left',33),('right',54)]:
                    hybrid[t,offset:offset+21]=0
                    for index,point in sample['hands'][side].items(): hybrid[t,offset+int(index)]=point
            variants['rendered_hands_payload_body']=hybrid
        for variant,data in variants.items():
            predictions.append(predict(data,label,variant))
        print(label,[(p['variant'],p['prediction']) for p in predictions if p['label']==label],flush=True)
    summaries=[]
    evaluated=[c['phrase'] for c in cases]
    for variant in sorted({p['variant'] for p in predictions}):
        rows=[p for p in predictions if p['variant']==variant]
        precisions=[]; recalls=[]; f1s=[]
        for label in evaluated:
            tp=sum(p['label']==label and p['prediction']==label for p in rows)
            fp=sum(p['label']!=label and p['prediction']==label for p in rows)
            fn=sum(p['label']==label and p['prediction']!=label for p in rows)
            precision=tp/(tp+fp) if tp+fp else 0
            recall=tp/(tp+fn) if tp+fn else 0
            precisions.append(precision);recalls.append(recall);f1s.append(2*precision*recall/(precision+recall) if precision+recall else 0)
        summaries.append(dict(variant=variant,n=len(rows),accuracy=100*np.mean([p['prediction']==p['label'] for p in rows]),
            top5=100*np.mean([p['label'] in p['top5'] for p in rows]),macro_precision=100*np.mean(precisions),macro_recall=100*np.mean(recalls),macro_f1=100*np.mean(f1s),
            median_runtime_ms=float(np.median([p['runtime_ms'] for p in rows]))))
    result=dict(summaries=summaries,predictions=predictions,labels_evaluated=evaluated,
        scope='Pilot, one source clip per class; unknown train overlap. Macro over selected true-label set; out-of-set predictions remain errors. Rendered variant uses actual hand strokes plus payload body, NOT whole-image recognition.',
        checkpoint_sha256=hashlib.sha256((HERE/'wlasl_stgcn.zip').read_bytes()).hexdigest())
    (audit_dir/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('--count',type=int,default=30)
    args=parser.parse_args()
    prepare(args.count) if args.stage=='prepare' else run()

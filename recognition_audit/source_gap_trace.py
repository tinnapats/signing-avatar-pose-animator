"""Read-only provenance and missing-hand trace; never infer sign boundaries."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import csv
import hashlib
import io
import json
import sqlite3
import zlib
from collections import Counter
from pathlib import Path
import numpy as np
import run_audit as audit
from repeat_original30 import production,sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'source_gap_trace_20261002'
SOURCE=HERE/'bridge_transition_20261002'
LOCAL=Path('C:/งาน/project_1/project_1/SLclean/SLclean')

def classify(start,end,n):
    if start==0 and end==n:return 'whole_clip'
    if start==0:return 'leading'
    if end==n:return 'trailing'
    return 'internal'

def spans(mask):
    i=0;out=[]
    while i<len(mask):
        if not mask[i]:i+=1;continue
        start=i
        while i<len(mask) and mask[i]:i+=1
        out.append(dict(start=start,end_exclusive=i,frames=i-start,kind=classify(start,i,len(mask))))
    return out

def main():
    assert not OUT.exists();OUT.mkdir();before=production()
    cases=[c for c in _local_json_loads((SOURCE/'review_inputs.json').read_text()) if c['variant']=='control'];assert len(cases)==12
    metadata={r['gloss']:r['instances'] for r in _local_json_loads((HERE/'kaggle_reference/WLASL_v0.3.json').read_text())}
    db=sqlite3.connect((ROOT/'SLclean.sqlite3').as_uri()+'?mode=ro',uri=True)
    columns=[r[1] for r in db.execute('PRAGMA table_info(clips)')];dbmeta=dict(db.execute('SELECT key,value FROM metadata'))
    (OUT/'protocol.json').write_text(json.dumps(dict(production=before,labels=[c['label'] for c in cases],
        source_hash=sha(SOURCE/'review_inputs.json'),sqlite_columns=columns,sqlite_metadata=dbmeta,
        note='No trimming, synthesis, new video downloads or sign-boundary labels. Whole-clip missing hand can be inactive or detection failure; cannot distinguish from coordinates alone.'),indent=2),encoding='utf-8')
    rows=[]
    for c in cases:
        label=c['label'];path,expected,compressed=db.execute('SELECT relative_path,sha256,csv_zlib FROM clips WHERE label=?',(label,)).fetchone()
        raw=zlib.decompress(compressed);assert hashlib.sha256(raw).hexdigest()==expected
        local=LOCAL/path
        localhash=sha(local) if local.exists() else None
        parsed=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))));frames={};times={}
        for r in parsed:
            f=int(r['frame']);times[f]=float(r['time_sec']);frames.setdefault(f,{}).setdefault(r['part'],{})[int(r['landmark_id'])]=r
        ids=sorted(frames);sides={}
        clips=audit.pipeline.discover_clips(ROOT/'SLclean.sqlite3')
        paths=audit.pipeline.resolve_selected_paths(clips,ROOT/'SLclean.sqlite3',[label+'.csv'],'')
        source30=audit.csv_array(audit.pipeline.concat_clips(paths,513,513,0,target_fps=30))
        sampled=[c['payload']['frames'][i] for i in c['frameIndices']];assert len(source30)==len(sampled)
        for side,offset,wrist in [('left',33,15),('right',54,16)]:
            rawmissing=[len(frames[f].get(side+'_hand',{}))<21 for f in ids]
            gaps=spans(rawmissing)
            for gap in gaps:
                fs=ids[gap['start']:gap['end_exclusive']]
                gap['first_frame_id']=fs[0];gap['last_frame_id']=fs[-1]
                gap['first_time_sec']=times[fs[0]];gap['last_time_sec']=times[fs[-1]]
                # Pose wrist visibility is only a diagnostic hint, not proof of hand visibility.
                gap['pose_wrist_visibility_ge_0_5_frames']=sum(float(frames[f].get('pose',{}).get(wrist,{}).get('visibility') or 0)>=.5 for f in fs)
            raw30=np.all(source30[:,offset:offset+21]==0,axis=(1,2))
            renderedmissing=np.array([not (f['hands'][side].get('score',0)>=.001 and sum(k.get('score',0)>0 for k in f['hands'][side].get('keypoints',[]))>=6) for f in sampled])
            payloadfull=c['payload']['frames']
            fullmissing=[not (f['hands'][side].get('score',0)>=.001 and sum(k.get('score',0)>0 for k in f['hands'][side].get('keypoints',[]))>=6) for f in payloadfull]
            sides[side]=dict(raw_frames=len(ids),raw_missing_frames=sum(rawmissing),raw_runs=gaps,
                payload_runs=spans(fullmissing),aligned30_frames=len(sampled),
                aligned30_both_missing=int(np.sum(raw30&renderedmissing)),
                aligned30_source_present_renderer_missing=int(np.sum(~raw30&renderedmissing)),
                aligned30_source_missing_renderer_present=int(np.sum(raw30&~renderedmissing)),
                aligned30_both_present=int(np.sum(~raw30&~renderedmissing)))
        rows.append(dict(label=label,sqlite_path=path,csv_sha256=expected,local_csv_path=str(local),
            local_csv_exact_match=localhash==expected,local_csv_exists=local.exists(),
            candidate_video_ids=[v['video_id'] for v in metadata[label]],verified_video_id=None,sides=sides))
        print(label,{s:(v['raw_missing_frames'],len(v['payload_runs'])) for s,v in sides.items()},flush=True)
    db.close();assert production()==before
    kinds=Counter(g['kind'] for r in rows for v in r['sides'].values() for g in v['payload_runs'])
    counts={k:sum(v[k] for r in rows for v in r['sides'].values()) for k in ['aligned30_both_missing','aligned30_source_present_renderer_missing','aligned30_source_missing_renderer_present','aligned30_both_present']}
    result=dict(rows=rows,payload_gap_types=dict(kinds),aligned30=counts,production_unchanged=True,
        verified_video_matches=0,local_csv_exact_matches=sum(r['local_csv_exact_match'] for r in rows))
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()

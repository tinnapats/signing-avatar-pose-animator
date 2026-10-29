"""Read-only dataset inventory and split feasibility; no training/download."""
import json,zipfile,collections
from pathlib import Path
import sqlite3
HERE=Path(__file__).resolve().parent;OUT=HERE/'training_data_inventory_20261005'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    assert not OUT.exists();OUT.mkdir()
    kaggle=read(HERE/'kaggle_reference/WLASL_v0.3.json')
    released=json.loads(zipfile.ZipFile(HERE/'wlasl_metadata.zip').read('splits/asl2000.json'))
    pairs={(r['gloss'],str(i['video_id'])):i for r in released for i in r['instances']}
    labels=read(HERE/'slgcn200_20261002/protocol.json')['labels'];assert len(labels)==200
    rows=[];candidates={};disagreements=[]
    for r in kaggle:
        if r['gloss'] not in labels:continue
        groups={s:[] for s in ['train','val','test']}
        for i in r['instances']:
            match=pairs.get((r['gloss'],str(i['video_id'])))
            if not match or match['split']!=i['split']:
                disagreements.append(dict(label=r['gloss'],video_id=i['video_id'],kaggle_split=i['split'],released_split=match.get('split') if match else None));continue
            if i['split'] in groups:groups[i['split']].append(i)
        for g in groups.values():g.sort(key=lambda x:str(x['video_id']))
        candidates[r['gloss']]=groups
        rows.append(dict(label=r['gloss'],**{s:len(g) for s,g in groups.items()},signers=len({x['signer_id'] for g in groups.values() for x in g})))
    # Global signer separation: preserve existing val/test roles; drop train
    # instances whose signer occurs anywhere in these evaluation splits.
    evalsigners={i['signer_id'] for g in candidates.values() for s in ['val','test'] for i in g[s]}
    for row in rows:
        row['train_after_eval_signer_exclusion']=sum(i['signer_id'] not in evalsigners for i in candidates[row['label']]['train'])
    local=[]
    for f in HERE.glob('*/manifest.json'):
        try:data=read(f)
        except (ValueError,UnicodeError):continue
        if not isinstance(data,list):continue
        for r in data:
            if not isinstance(r,dict) or not r.get('local_video'):continue
            path=Path(r['local_video']);m=r.get('metadata',{})
            local.append(dict(label=r.get('word'),video_id=m.get('video_id'),split=m.get('split'),exists=path.is_file(),path=str(path),manifest=str(f)))
    unique={str(x['video_id']):x for x in local if x['exists'] and x['video_id'] is not None}
    csvdir=Path('C:/งาน/project_1/project_1/SLclean/SLclean')
    csvfiles=list(csvdir.glob('*.csv'))
    tables={}
    with sqlite3.connect((HERE.parent/'SLclean.sqlite3').as_uri()+'?mode=ro',uri=True) as con:
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            safe=name.replace('"','""');tables[name]=dict(columns=[x[1] for x in con.execute(f'PRAGMA table_info("{safe}")')],count=con.execute(f'SELECT COUNT(*) FROM "{safe}"').fetchone()[0])
    eligible=[r['label'] for r in rows if r['train']>=3 and r['val']>=1 and r['test']>=2]
    summary=dict(words=200,metadata_instances_by_split={s:sum(r[s] for r in rows) for s in ['train','val','test']},minimum_by_split={s:min(r[s] for r in rows) for s in ['train','val','test']},eligible_3train_1val_2test=len(eligible),eligible_labels=eligible,split_disagreements=len(disagreements),local_manifest_unique_videos=len(unique),local_by_split=dict(collections.Counter(x['split'] for x in unique.values())),local_training_selected200=sum(x['split']=='train' and x['label'] in labels for x in unique.values()),selected200_local_videos=sum(x['label'] in labels for x in unique.values()),slclean_csv_count=len(csvfiles),sqlite=tables,train_after_global_signer_exclusion=sum(r['train_after_eval_signer_exclusion'] for r in rows),labels_with_3train_after_global_signer_exclusion=sum(r['train_after_eval_signer_exclusion']>=3 for r in rows),scope='Metadata eligibility only. Remote files not yet checked/downloaded; manifests cover audited local video locations, not entire computer. Existing SQLite clip names do not prove video_id pairing. No supervised training or final-test claim.')
    result=dict(summary=summary,per_word=rows,candidates=candidates,split_disagreements=disagreements,local_manifest_videos=local)
    (OUT/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()

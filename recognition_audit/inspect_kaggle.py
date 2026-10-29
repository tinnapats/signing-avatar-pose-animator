"""Fetch small metadata only; inspect SQLite read-only; do not infer clip identity."""
import hashlib
import io
import json
import sqlite3
import urllib.request
import urllib.parse
import zipfile
import zlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'kaggle_reference'
BASE='https://www.kaggle.com/api/v1/datasets'

def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'signing-avatar-research-metadata/1.0'})
    with urllib.request.urlopen(req,timeout=60) as response:
        data=response.read(25_000_001)
    if len(data)>25_000_000:raise ValueError('Metadata size limit exceeded')
    return data

def main():
    OUT.mkdir(exist_ok=True)
    view=json.loads(fetch(BASE+'/view/sttaseen/wlasl2000-resized'))
    assert view['currentVersionNumber']==1, 'Review changed dataset version first'
    (OUT/'dataset_info.json').write_text(json.dumps(view,indent=2),encoding='utf-8')
    evidence={}
    for name in ['WLASL_v0.3.json','nslt_2000.json']:
        url=BASE+'/download/sttaseen/wlasl2000-resized/'+urllib.parse.quote('wlasl-complete/'+name,safe='')+'?datasetVersionNumber=1'
        raw=fetch(url)
        if raw.startswith(b'PK'):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                matches=[n for n in archive.namelist() if Path(n).name==name]
                assert len(matches)==1
                raw=archive.read(matches[0])
        json.loads(raw)
        (OUT/name).write_bytes(raw)
        evidence[name]=dict(url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        print('downloaded',name,len(raw),flush=True)
    metadata=json.loads((OUT/'WLASL_v0.3.json').read_bytes())
    by_label={r['gloss'].lower():r['instances'] for r in metadata}
    with sqlite3.connect((ROOT/'SLclean.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        columns=[r[1] for r in db.execute('PRAGMA table_info(clips)')]
        local=db.execute('SELECT label,relative_path,frame_count,columns_json FROM clips').fetchall()
        database_meta=dict(db.execute('SELECT key,value FROM metadata'))
        example=zlib.decompress(db.execute('SELECT csv_zlib FROM clips LIMIT 1').fetchone()[0]).splitlines()[:2]
    mappings=[]
    for label,path,frames,cols in local:
        candidates=by_label.get(label.lower(),[])
        mappings.append(dict(label=label,local_path=path,local_frames=frames,match_status='label_only_not_verified' if candidates else 'no_label_match',candidates=[{k:c.get(k) for k in ['video_id','signer_id','split','frame_start','frame_end','fps']} for c in candidates]))
    selected={r['word'] for r in json.loads((HERE/'diagnosis/review_cases.json').read_text())}
    summary=dict(version=1,dataset_bytes=view.get('totalBytes'),metadata_labels=len(metadata),metadata_instances=sum(len(r['instances']) for r in metadata),sqlite_clips=len(local),labels_matched=sum(bool(m['candidates']) for m in mappings),verified_video_matches=0,sqlite_columns=columns,sqlite_metadata=database_meta,csv_example=[r.decode('utf-8-sig') for r in example],evidence=evidence)
    (OUT/'inspection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'label_candidates.json').write_text(json.dumps(mappings,indent=2),encoding='utf-8')
    focus=[m for m in mappings if m['label'] in selected]
    (OUT/'review_candidates.json').write_text(json.dumps(focus,indent=2),encoding='utf-8')
    lines=['# ตรวจ Kaggle WLASL-2000 Resized','',f"เวอร์ชัน 1; ขนาดตาม Kaggle {summary['dataset_bytes']:,} bytes",f"Metadata: {summary['metadata_labels']} คำ / {summary['metadata_instances']} instances (ไม่ใช่จำนวนวิดีโอที่ยืนยันว่ามีใน Kaggle)",f"SQLite: {len(local)} คลิป; ชื่อคำตรง metadata {summary['labels_matched']} คลิป; ยืนยันคลิปเดียวกันได้ 0 คลิป",'', 'ดาวน์โหลดเฉพาะ JSON metadata และ split ไม่ได้โหลดวิดีโอทั้งชุด ไม่เปลี่ยนฐานข้อมูล', '', '| คำที่ตรวจ | จำนวนวิดีโอที่เป็นไปได้จาก metadata |','|---|---:|']
    lines += [f"| {m['label']} | {len(m['candidates'])} |" for m in focus]
    lines += ['', '## ข้อจำกัดและทางต่อ', '- ชื่อคำเดียวกันไม่ยืนยันว่าเป็นคลิปเดียวกัน ห้ามเลือกวิดีโอแรกเป็น ground truth', '- เก็บรหัส video_id/signer_id/split ของ candidate ไว้เพื่อค้นหาต้นฉบับต่อ', '- ต้องหาไฟล์ mapping จากขั้นตอนสร้าง SLclean หรือหลักฐานที่ยืนยันวิดีโอต้นทาง', '- Metadata มีหลาย instances และอาจระบุวิดีโอที่ไม่ได้อยู่ในแพ็ก Kaggle ต้องตรวจไฟล์จริงก่อนใช้', '- split ของ Kaggle/WLASL ยังต้องเทียบกับ split ของ checkpoint ที่ใช้อยู่ ไม่ถือว่าตรงโดยอัตโนมัติ', '- เงื่อนไขที่หน้า dataset ระบุ: งานวิชาการ/คำนวณ ภายใต้ C-UDA ไม่อนุญาตเชิงพาณิชย์ ต้องอ่านเงื่อนไขก่อนใช้หรือเผยแพร่วิดีโอ', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

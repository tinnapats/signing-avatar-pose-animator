
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'missing_imputation_20261008';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B.parent));import run_audit as audit
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
assert not (O/'results.json').exists(),'Preserve completed results'
names={'zero_baseline':'ค่าศูนย์เดิม','nearest':'ใช้ท่าที่พบใกล้ที่สุด','linear':'แทรกพิกัดระหว่างสองท่า','bone_angles':'แทรกทิศกระดูกและความยาว'}
save(O/'protocol.json',dict(names=names,source='rest_gate_flow/rest_roi3',rules='Split each side into segments bounded by rest gate. Preserve observed/tracked values and gated zeros. Impute only missing hand blocks within each segment with at least one anchor; no label or Renderer reference used. Edges use nearest anchor. Segments without anchors remain explicitly unresolved. No maximum gap for exploratory comparison; record distance to nearest evidence. Offline uses future frames. Bone interpolation shortest angle in2D, linear lengths, wrist linear. Camera metadata used only for common model normalization.',state_codes={'0':'unresolved missing','1':'existing detection or prior tracking','2':'rest suppressed','3':'estimated'},fps=30))
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
def bone(a,b,t):
 out=np.zeros_like(a);out[0]=(1-t)*a[0]+t*b[0]
 for pa,ch in edges:
  v=a[ch]-a[pa];w=b[ch]-b[pa];aa=np.arctan2(v[1],v[0]);bb=np.arctan2(w[1],w[0]);angle=aa+t*((bb-aa+np.pi)%(2*np.pi)-np.pi);length=(1-t)*np.linalg.norm(v)+t*np.linalg.norm(w);out[ch]=out[pa]+length*np.array([np.cos(angle),np.sin(angle)])
 return out
# Minimal independent sanity checks: preserve endpoints and rotating unit bones do not collapse.
a=np.array([[float(i),0.] for i in range(21)]);b=a[:,::-1].copy();assert np.allclose(bone(a,b,0),a) and np.allclose(bone(a,b,1),b)
refs=read(B/'mixed_reference.json');model,norm,vocab=audit.load_model();rows=[];geo=[];counts={};prior=read(B/'rest_gate_flow/results.json')['predictions']
for c in refs:
 z=np.load(B/'rest_gate_flow'/(c['id']+'.npz'));base=z['rest_roi3'];gate=z['gate'];n=len(base)
 for variant in names:
  data=base.copy();state=np.zeros((n,2),np.uint8);distance=np.full((n,2),-1,np.int32)
  for side,off in enumerate([33,54]):
   observed=np.any(base[:,off:off+21]!=0,axis=(1,2));state[observed,side]=1;state[gate[:,side],side]=2;distance[observed,side]=0
   if variant=='zero_baseline':continue
   starts=np.flatnonzero(~gate[:,side]&np.r_[True,gate[:-1,side]])
   for start in starts:
    end=start
    while end<n and not gate[end,side]:end+=1
    anchors=np.flatnonzero(observed[start:end])+start
    if not len(anchors):continue
    for i in range(start,end):
     if observed[i]:continue
     idx=int(np.searchsorted(anchors,i));l=int(anchors[max(0,idx-1)]);r=int(anchors[min(idx,len(anchors)-1)]);near=l if abs(l-i)<=abs(r-i) else r;distance[i,side]=abs(near-i)
     if variant=='nearest' or l==r:data[i,off:off+21]=base[near,off:off+21]
     else:
      t=(i-l)/(r-l);aa=base[l,off:off+21];bb=base[r,off:off+21];data[i,off:off+21]=(1-t)*aa+t*bb if variant=='linear' else bone(aa,bb,t)
     state[i,side]=3
  assert np.array_equal(data[:,:33],base[:,:33])
  for side,off in enumerate([33,54]):assert np.array_equal(data[state[:,side]==1,off:off+21],base[state[:,side]==1,off:off+21]);assert not np.any(data[gate[:,side],off:off+21])
  np.savez_compressed(O/(c['id']+'_'+variant+'.npz'),points=data,state=state,distance_frames=distance)
  cnt=counts.setdefault(variant,dict(estimated=0,unresolved=0,rest=0,distant_over10=0));cnt['estimated']+=int((state==3).sum());cnt['unresolved']+=int((state==0).sum());cnt['rest']+=int((state==2).sum());cnt['distant_over10']+=int(((state==3)&(distance>10)).sum())
  for i,r in enumerate(c['samples']):
   for si,(s,off) in enumerate([('left',33),('right',54)]):
    if len(r['hands'][s])==21 and state[i,si] in [1,3]:
     gt=np.array([r['hands'][s][str(j)] for j in range(21)]);geo.append(dict(variant=variant,estimated=bool(state[i,si]==3),squared=float(np.mean(np.sum((data[i,off:off+21]-gt)**2,axis=1)))))
   origin,ex,ey=np.array(r['affine'])*r['pixelRatio'];inv=np.linalg.inv(np.stack([ex-origin,ey-origin],axis=1));valid=np.any(data[i]!=0,axis=1);data[i,valid]=(data[i,valid]-origin)@inv.T
  x=norm({'frames':torch.from_numpy(data[:,audit.PICK,:].transpose(2,0,1).copy())})['frames'].unsqueeze(0);assert torch.isfinite(x).all()
  with torch.inference_mode():p=model(x)[0].softmax(0)
  top=p.topk(5);row=dict(label=c['label'],variant=variant,prediction=vocab[int(top.indices[0])],top5=[vocab[int(t)] for t in top.indices]);rows.append(row)
  if variant=='zero_baseline':assert row['prediction']==next(r['prediction'] for r in prior if r['label']==c['label'] and r['variant']=='rest_roi3')
 print('scored',c['label'],flush=True)
summary={}
for v in names:
 rr=[r for r in rows if r['variant']==v];ps=[];rs=[];fs=[]
 for l in read(B/'protocol.json')['labels']:
  tp=sum(r['label']==l and r['prediction']==l for r in rr);fp=sum(r['label']!=l and r['prediction']==l for r in rr);p=tp/(tp+fp) if tp+fp else 0;ps.append(p);rs.append(tp);fs.append(2*p*tp/(p+tp) if p+tp else 0)
 gg=[g for g in geo if g['variant']==v];eg=[g for g in gg if g['estimated']];summary[v]=dict(correct=sum(r['label']==r['prediction'] for r in rr),precision=100*np.mean(ps),recall=100*np.mean(rs),f1=100*np.mean(fs),reference_matched=len(gg),rmse=float(np.sqrt(np.mean([g['squared'] for g in gg]))),estimated_reference=len(eg),estimated_rmse=float(np.sqrt(np.mean([g['squared'] for g in eg]))) if eg else None,**counts[v])
for f,h in read(B/'protocol.json')['production'].items():assert hashlib.sha256((B.parent.parent/f).read_bytes()).hexdigest()==h
save(O/'results.json',dict(summary=summary,predictions=rows,production_unchanged=True))
lines=['# ทดลองแทนค่าศูนย์ของมือที่ตรวจไม่พบ','', '8 ตุลาคม2026: 30คำเดิม1964เฟรม30FPS ตรึงลำตัวและกฎคัดมือพัก ใช้จุดที่อ่านจากภาพเป็นฐาน จุดRendererใช้วัดผลเท่านั้น; ยังคงใช้camera metadataเพื่อจัดพิกัดเข้าโมเดล','', '| วิธี | ถูก/30 | Macro-F1 | เติมมือ×เฟรม | ยังไม่มีหลักฐานให้เติม | RMSEของจุดเติมที่มีreference(px) |','|---|---:|---:|---:|---:|---:|']
for v,s in summary.items():lines.append(f"| {names[v]} | {s['correct']}/30 | {s['f1']:.2f}% | {s['estimated']} | {s['unresolved']} | {s['estimated_rmse']} |")
lines+=['','จุดที่ตรวจพบ/ติดตามได้เดิมคงเดิมทุกจุด กฎคัดมือพักไม่ถูกเติมกลับ แต่ละnpzมีpoints,state,distance_frames ระบุจุดประมาณแยกจากหลักฐานจริง','', 'หากทั้งช่วงไม่มีจุดมือเป็นanchor ไม่สร้างท่าจากลำตัวหรือคำเฉลย ยังคงmissingไว้เป็นศูนย์ตามข้อจำกัดinputโมเดลเดิม พร้อมstate=0 การทดลองนี้จึงลดค่าศูนย์ได้บางส่วน ไม่ได้กำจัดทั้งหมด หากต้องการห้ามศูนย์ทุกกรณีต้องเพิ่มทางออกabstain/ข้อมูลไม่พอ หรือโมเดลรองรับmask','', 'วิธีnearestอาจทำให้มือค้าง; linearอาจทำให้กระดูกหดระหว่างท่า; bone_anglesคุมความยาวตามค่าระหว่างanchorใน2D แต่ไม่รับรองมุมข้อต่อ/3Dหรือการคงรายละเอียดที่หายไป ช่วงไม่มีหลักฐานยาวเสี่ยงผิดท่า บันทึกdistant_over10ไว้ในresults.json','', 'เป็นofflineใช้เฟรมอนาคต ผลชุดพัฒนาเดิมไม่ใช่held-out และไม่มีการแก้production คะแนนที่ดีขึ้นยังไม่ยืนยันความถูกต้องASL','', 'รายละเอียดPrecision/Recall/F1และคำทำนายรายคำอยู่results.json']
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(summary,indent=2))

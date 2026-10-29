import sys,importlib.util,time
from pathlib import Path
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('base',Path(__file__).with_name('rtm-avatar.py'));base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
np=base.np;O=base.O;B=base.B;read=base.read;save=base.save;sha=base.sha
def main():
 cv2,rt,det=base.setup();hand=base.load('hand',rt,det);out=O/'hand_crop';out.mkdir(exist_ok=True)
 protocol=dict(body='rtmw_det',rules='Predefined exploratory body choice, not selected by recognition score. Wrist/elbow confidence>=.3; center=wrist+.25*(wrist-elbow), square side=clip(2.2*forearm,96,256); RTMPose-hand adds its default bbox padding1.25. Hand mean score and wrist score>=.3; predicted wrist within .65 forearm of body wrist; two crop wrists within20px => reject less body-consistent. No temporal repair/smoothing. Same original PNG/timing.',manifest_sha256=sha(B/'capture_manifest.json'))
 if not (out/'protocol.json').exists():save(out/'protocol.json',protocol)
 else:assert read(out/'protocol.json')==protocol
 for c in read(B/'capture_manifest.json'):
  src=O/(c['id']+'.npz');target=out/(c['id']+'.npz');meta=out/(c['id']+'.json')
  if target.exists():assert meta.exists() and sha(target)==read(meta)['sha256'];continue
  if not src.exists():break
  assert sha(src)==read(O/(c['id']+'.json'))['sha256'];z=np.load(src);body=z['rtmw_det'];scores=z['rtmw_det_scores'];pts=[];conf=[];boxes=[];accepted=[];ms=[]
  for i,f in enumerate(c['frames']):
   p=B/'frames'/c['id']/f['file'];assert sha(p)==f['sha256'];im=base.image(cv2,p);pp=np.zeros((2,21,2),np.float32);ss=np.zeros((2,21),np.float32);bb=np.zeros((2,4),np.float32);vv=np.zeros(2,bool);dist=np.full(2,np.inf);start=time.perf_counter()
   for side,(w,e) in enumerate([(9,7),(10,8)]):
    if scores[i,w]<.3 or scores[i,e]<.3:continue
    wrist=body[i,w];delta=wrist-body[i,e];length=float(np.linalg.norm(delta))
    if length<10:continue
    center=wrist+.25*delta;size=np.clip(2.2*length,96,256);bbox=np.r_[center-size/2,center+size/2];bb[side]=bbox
    p,s=hand(im,bboxes=[bbox]);pp[side]=p[0];ss[side]=s[0];dist[side]=np.linalg.norm(p[0,0]-wrist)
    vv[side]=s[0].mean()>=.3 and s[0,0]>=.3 and dist[side]<=.65*length
   if vv.all() and np.linalg.norm(pp[0,0]-pp[1,0])<20:vv[int(np.argmax(dist))]=False
   pts.append(pp);conf.append(ss);boxes.append(bb);accepted.append(vv);ms.append((time.perf_counter()-start)*1000)
  np.savez_compressed(target,points=pts,scores=conf,boxes=boxes,accepted=accepted,times=z['times'],ms=ms);save(meta,dict(label=c['label'],sha256=sha(target),frames=len(pts),accepted=int(np.sum(accepted)),ms_per_frame=float(np.mean(ms))))
  print('crop',c['id'],c['label'],'accepted',np.sum(accepted),'/',2*len(pts),flush=True)
 print('Available clips processed',flush=True)
if __name__=='__main__':main()

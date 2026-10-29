"""Six preselected pixel-only frames; staged official SDPose weights, no production edits."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import os, sys, json, time, hashlib, gc, traceback, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'mpl-cache'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, str(ROOT / 'source'))
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import load_file
from models.HeatmapHead import get_heatmap_head
from models.ModifiedUNet import Modified_forward

B = Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
R = B / 'rtm_trial_20261008'
parser=argparse.ArgumentParser()
parser.add_argument('--full-three',action='store_true')
parser.add_argument('--next-three',action='store_true')
args=parser.parse_args()
assert not (args.full_three and args.next_three)
OUT = ROOT / ('confirmation3' if args.next_three else ('tracking3' if args.full_three else 'pilot6'))
if (OUT/'protocol.json').exists():
    raise RuntimeError('Existing output must be preserved; choose a fresh experiment directory.')
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
records = []
protocol = {'status': 'started', 'pixel_only': True, 'recognition_run': False,
    'selection': 'ids 000,001,002; frame 0 and floor(T/2), fixed before inference',
    'model_revision': '0c8153e39458bc8ae87fb68790c141bf0f03c11b',
    'input_wh': [768,1024], 'unet_dtype': 'float16', 'vae_dtype': 'float32',
    'text_encoder_dtype': 'float32', 'decoder_dtype': 'float32',
    'timestep':999, 'seed_each_frame':0, 'flip_test':False,
    'crop': 'cached YOLOX person box, clipped integers, no reference keypoints used',
    'note': 'Staged inference mirrors official single_infer. FP16 UNet is a memory adaptation, not FP32 equivalence verified.'}
if args.full_three:
    protocol['selection']='All frames of ids 000,001,002 at original captured timestamps; no resampling'
if args.next_three:
    protocol['selection']='Next three manifest entries 003,004,005, fixed by order, not recognition scores; all original frames. New to SDPose experiment, not an independent untouched test set.'

def save():
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (OUT/'results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

def checked(info, name):
    errors = {k:info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') if info.get(k)}
    if errors: raise RuntimeError(f'{name} weights mismatch: {errors}')

def clean():
    gc.collect()
    torch.cuda.empty_cache()

try:
    assert _local_json_loads((ROOT/'weights_manifest.json').read_text())['status']=='downloaded_size_checked_and_hashed'
    assert torch.cuda.is_available()
    manifest = _local_json_loads((B/'capture_manifest.json').read_text())
    preprocess=transforms.Compose([transforms.Resize((1024,768)),transforms.ToTensor(),
                                  transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    for clip in (manifest[3:6] if args.next_three else manifest[:3]):
        cached=np.load(R/f"{clip['id']}.npz")
        for i in (range(len(clip['frames'])) if (args.full_three or args.next_three) else (0,len(clip['frames'])//2)):
            fr=clip['frames'][i]; path=B/'frames'/clip['id']/fr['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==fr['sha256']
            im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if im.shape[2]==4:
                a=im[:,:,3:4].astype(np.float32)/255
                im=np.rint(im[:,:,:3]*a+255*(1-a)).astype(np.uint8)
            h,w=im.shape[:2]
            box=np.asarray(cached['boxes'][i]).reshape(-1)[:4]
            assert np.isfinite(box).all()
            x1,y1,x2,y2=map(int,box)
            x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            assert x2>x1 and y2>y1
            tag=f"{clip['id']}_{i:04d}"
            cv2.imwrite(str(OUT/f'{tag}_input.png'),im)
            records.append({'id':clip['id'],'label':clip['label'],'frame':i,'sourceIndex':fr['sourceIndex'],
                'time':fr['time'],'sha256':fr['sha256'],'tag':tag,'box':[x1,y1,x2,y2]})
            torch.save(preprocess(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop((x1,y1,x2,y2))).unsqueeze(0),OUT/f'{tag}_input.pt')
    save()
    weights=str(ROOT/'weights')
    print('Loading text encoder on CPU',flush=True)
    tokenizer=CLIPTokenizer.from_pretrained(weights,subfolder='tokenizer',local_files_only=True)
    text,info=CLIPTextModel.from_pretrained(weights,subfolder='text_encoder',local_files_only=True,output_loading_info=True,low_cpu_mem_usage=True)
    checked(info,'text'); text.eval()
    ids=tokenizer('',padding='do_not_pad',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt').input_ids
    with torch.inference_mode(): embedding=text(ids)[0].detach()
    del text,tokenizer; clean()
    print('Loading VAE float32',flush=True)
    vae,info=AutoencoderKL.from_pretrained(weights,subfolder='vae',local_files_only=True,output_loading_info=True,low_cpu_mem_usage=True)
    checked(info,'vae'); vae.eval().to('cuda')
    for row in records:
        print('Encode',row['tag'],flush=True)
        x=torch.load(OUT/f"{row['tag']}_input.pt",weights_only=True)
        start=time.perf_counter(); torch.manual_seed(0)
        device=str(next(vae.parameters()).device)
        try:
            with torch.inference_mode(): latent=vae.encode(x.to(device)).latent_dist.sample()*.18215
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            vae.to('cpu'); clean(); torch.manual_seed(0); device='cpu'
            with torch.inference_mode(): latent=vae.encode(x).latent_dist.sample()*.18215
        assert torch.isfinite(latent).all()
        torch.save(latent.cpu(),OUT/f"{row['tag']}_latent.pt")
        row.update(vae_seconds=time.perf_counter()-start,vae_device=device)
        del latent,x; clean(); save()
    del vae; clean()
    print('Loading UNet float16, strict loading diagnostics',flush=True)
    unet,info=UNet2DConditionModel.from_pretrained(weights,subfolder='unet',local_files_only=True,
        torch_dtype=torch.float16,low_cpu_mem_usage=True,output_loading_info=True,
        class_embed_type='projection',projection_class_embeddings_input_dim=4)
    checked(info,'unet'); unet=Modified_forward(unet,'wholebody').eval().to('cuda')
    decoder=get_heatmap_head('wholebody')
    decoder.load_state_dict(load_file(str(ROOT/'weights/decoder/decoder.safetensors')),strict=True)
    decoder.eval()
    emb=embedding.to(device='cuda',dtype=torch.float16)
    task=torch.tensor([[1.,0.]],device='cuda')
    task=torch.cat([torch.sin(task),torch.cos(task)],dim=-1).half()
    for row in records:
        print('Predict',row['tag'],flush=True)
        z=torch.load(OUT/f"{row['tag']}_latent.pt",weights_only=True).to(device='cuda',dtype=torch.float16)
        torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        with torch.inference_mode():
            feat=unet(z,torch.tensor(999,device='cuda').long(),emb,class_labels=task,
                return_dict=False,return_decoder_feats=True)
            feat_cpu=feat.float().cpu()
        torch.cuda.synchronize()
        row['unet_seconds']=time.perf_counter()-start
        row['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        assert torch.isfinite(feat_cpu).all()
        del feat,z; unet._decoder_feats=[]; clean()
        with torch.inference_mode(): pred=decoder.predict((feat_cpu,),None,test_cfg={'flip_test':False})[0]
        kp=np.asarray(pred.keypoints[0]); scores=np.asarray(pred.keypoint_scores[0])
        assert kp.shape==(133,2) and np.isfinite(kp).all() and np.isfinite(scores).all()
        x1,y1,x2,y2=row['box']; kp=kp*np.array([(x2-x1)/768,(y2-y1)/1024])+[x1,y1]
        row.update(keypoints=kp.tolist(),scores=scores.tolist())
        np.savez_compressed(OUT/f"{row['tag']}.npz",keypoints=kp,scores=scores)
        im=cv2.imread(str(OUT/f"{row['tag']}_input.png"))
        edges=[(5,7),(7,9),(6,8),(8,10),(5,6)]
        for base in (91,112):
            for chain in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)):
                edges.extend((base+a,base+b) for a,b in zip(chain,chain[1:]))
        for a,b in edges:
            if min(scores[a],scores[b])>=.3: cv2.line(im,tuple(np.rint(kp[a]).astype(int)),tuple(np.rint(kp[b]).astype(int)),(0,160,0),2)
        for j in list(range(17))+list(range(91,133)):
            if scores[j]>=.3: cv2.circle(im,tuple(np.rint(kp[j]).astype(int)),3,(0,0,255),-1)
        cv2.putText(im,f"SDPose {row['label']} frame {row['frame']} threshold .30",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2)
        cv2.imwrite(str(OUT/f"{row['tag']}_overlay.png"),im)
        save()
    protocol['status']='three_clip_inference_completed' if (args.full_three or args.next_three) else 'six_frame_inference_completed'; save()
    print('PILOT COMPLETE',flush=True)
except Exception:
    protocol['status']='failed'; protocol['error']=traceback.format_exc(); save()
    print(protocol['error'],flush=True)
    raise

"""Evaluate the isolated four-cell fallback ablation, not ASL accuracy."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import cv2
import imageio_ffmpeg

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE/'i3d_avatar/deps'))
from PIL import Image,ImageDraw,ImageOps
OUT=HERE/'fallback_trial'
VARIANTS=['baseline','scale_only','cap_only','combined']
PAIRS=[('cap_only','baseline'),('combined','scale_only')]
PRIOR_MAPPING={'S1A1':'baseline','S50A1':'scale_only'}
CAP_VARIANTS=['cap_only','combined']

def main():
    diagnostic=json.loads((HERE/'method_review/diagnostics.json').read_text())
    actual={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in diagnostic['production_hashes']}
    assert actual==diagnostic['production_hashes'],'Production changed'
    captures=json.loads((OUT/'rendered.json').read_text())
    lookup={(r['variant'],r['phrase']):r['samples'] for r in captures}
    prior=json.loads((HERE/'method_review/rendered.json').read_text())
    for r in prior:
        if r['variant'] not in PRIOR_MAPPING:continue
        v=PRIOR_MAPPING[r['variant']]
        samples=lookup[v,r['phrase']]
        assert len(samples)==len(r['samples'])
        for a,b in zip(samples,r['samples']):assert a['hands']==b['hands']
    rows=[]
    for d in diagnostic['rows']:
        word=d['word'];stats={}
        for candidate,control in PAIRS:
            for a,b in zip(lookup[candidate,word],lookup[control,word]):
                assert a['hands']==b['hands'],'Detailed joints changed'
        for v in VARIANTS:
            samples=lookup[v,word];records=[r for f in samples for r in f['fallback'].values() if r['nativePaths']]
            eligible=[r for r in records if r['limit'] and r['after'] is not None]
            if v in CAP_VARIANTS:
                assert all(r['after']<=r['limit']+1e-6 for r in eligible)
            ratios=[r['after']/r['limit'] for r in eligible]
            # Boundary image change is descriptive: smaller area alone can reduce it.
            all_steps=[];boundary_steps=[];previous=None
            folder=OUT/'frames'/v/word
            for i,f in enumerate(samples):
                im=cv2.imread(str(folder/f'{i:04d}.png'));assert im is not None
                if previous is not None:
                    change=float(sum(cv2.mean(cv2.absdiff(im,previous))[:3])/(3*255))
                    all_steps.append(change)
                    if any(bool(f['fallback'][s]['nativePaths']) != bool(samples[i-1]['fallback'][s]['nativePaths']) for s in ['left','right']):boundary_steps.append(change)
                previous=im
            native_steps=[];handoff_steps=[]
            for previous_frame,f in zip(samples,samples[1:]):
                for side in ['left','right']:
                    a,b=previous_frame['fallback'][side],f['fallback'][side]
                    step=abs(a['factor']-b['factor'])
                    if a['nativePaths'] and b['nativePaths']:native_steps.append(step)
                    elif bool(a['nativePaths'])!=bool(b['nativePaths']):handoff_steps.append(step)
            if v=='chain_slew':assert max(native_steps,default=0)<=.10000001
            stats[v]=dict(native_hand_frames=len(records),eligible_hand_frames=len(eligible),
                uncalibrated_hand_frames=len(records)-len(eligible),clamped_hand_frames=sum(r['factor']<1 for r in records),
                minimum_factor=min((r['factor'] for r in records),default=1),
                maximum_native_factor_step=max(native_steps,default=0),maximum_handoff_factor_step=max(handoff_steps,default=0),
                above_limit_hand_frames=sum(r['after']>r['limit']+1e-6 for r in eligible),
                maximum_radius_over_limit=max(ratios,default=None),
                mean_boundary_pixel_change=float(np.mean(boundary_steps)) if boundary_steps else None,
                boundary_transitions=len(boundary_steps),pixel_change_p95=float(np.quantile(all_steps,.95)))
        # Same crop and timestamps for all four variants, never auto-fit each tile.
        times=d['review_frames'];w=200;h=220
        sheet=Image.new('RGB',(w*len(times),h*4+30),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,8),word+' | rows: '+' / '.join(VARIANTS),(0,0,0))
        boxes=[]
        for v in VARIANTS:
            for i in times:
                im=Image.open(OUT/'frames'/v/word/f'{i:04d}.png').convert('RGB');a=np.asarray(im)
                y,x=np.where(np.any(a<235,axis=2))
                if len(x):boxes.append((x.min(),y.min(),x.max()+1,y.max()+1))
        crop=(min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes))
        for row,v in enumerate(VARIANTS):
            for col,i in enumerate(times):
                im=Image.open(OUT/'frames'/v/word/f'{i:04d}.png').convert('RGB').crop(crop)
                tile=ImageOps.contain(im,(w-8,h-24));sheet.paste(tile,(col*w+(w-tile.width)//2,30+row*h+24))
                draw.text((col*w+3,30+row*h+3),f'{v} {i/30:.2f}s',(0,0,0))
        sheet.save(OUT/f'{word}_comparison.jpg')
        rows.append(dict(word=word,frames=d['frames'],variants=stats))
    (OUT/'videos').mkdir(exist_ok=True)
    for v in VARIANTS:
        target=OUT/'videos'/f'crazy_{v}.mp4'
        if not target.exists():subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-framerate','30','-i',str(OUT/'frames'/v/'crazy'/'%04d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(target)],check=True)
    result=dict(production_unchanged=True,prior_joint_capture_exact=True,detailed_joints_unchanged=True,
                cap_verified=True,strict_cap_variants=CAP_VARIANTS,scope='5 known development failures; not a new held-out validation',rows=rows)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

"""Scientific contact sheets and videos for inspecting known regressions."""
import json
from pathlib import Path
import subprocess
import sys
import hashlib
import numpy as np
import cv2
import imageio_ffmpeg
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'i3d_avatar/deps'))
from PIL import Image,ImageOps,ImageDraw
OUT=HERE/'method_review'

def main():
    data=json.loads((OUT/'diagnostics.json').read_text());(OUT/'videos').mkdir(exist_ok=True)
    old={(r['variant'],r['phrase']):r['samples'] for r in json.loads((HERE/'heldout_scale_trial/comparison/rendered.json').read_text())}
    fresh=json.loads((OUT/'rendered.json').read_text())
    for r in fresh:
        prior=old[r['variant'],r['phrase']];assert len(prior)==len(r['samples'])
        for a,b in zip(prior,r['samples']):
            for side in ['left','right']:
                assert a['hands'][side].keys()==b['hands'][side].keys()
                for k,p in a['hands'][side].items():assert np.linalg.norm(np.array(p)-b['hands'][side][k])<1e-6
    for row in data['rows']:
        word=row['word'];boxes=[]
        for variant in ['S1A1','S50A1']:
            folder=OUT/'frames'/variant/word;paths=sorted(folder.glob('*.png'));assert len(paths)==row['frames']
            for path in paths:
                a=np.asarray(Image.open(path).convert('RGB'),dtype=float);r,g,b=[a[:,:,i] for i in range(3)]
                y,x=np.where((r>180)&(r>g*1.1)&(r>b*1.1))
                if len(x):boxes.append((int(x.min()),int(y.min()),int(x.max()),int(y.max())))
            target=OUT/'videos'/(word+'_'+variant+'.mp4')
            if not target.exists():subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-framerate','30','-i',str(folder/'%04d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(target)],check=True)
        l=min(b[0] for b in boxes)-25;t=min(b[1] for b in boxes)-25;r=max(b[2] for b in boxes)+25;b=max(b[3] for b in boxes)+25
        crop=(l,t,r,b);times=row['review_frames'];w=220;h=250
        sheet=Image.new('RGB',(w*len(times),h*3+32),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,8),word+' | source / baseline / S50A1 | fixed shared avatar crop',(0,0,0))
        cap=cv2.VideoCapture(row['source_video']);fps=cap.get(cv2.CAP_PROP_FPS)
        for column,index in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_FRAMES,round(index/30*fps));ok,frame=cap.read();assert ok
            original=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            ims=[original]+[Image.open(OUT/'frames'/v/word/(str(index).zfill(4)+'.png')).convert('RGB').crop(crop) for v in ['S1A1','S50A1']]
            for ri,im in enumerate(ims):
                tile=ImageOps.contain(im,(w-8,h-28));x=column*w+(w-tile.width)//2;y=32+ri*h+24
                sheet.paste(tile,(x,y));draw.text((column*w+6,32+ri*h+5),['source','baseline','S50A1'][ri]+f' t={index/30:.2f}s',(0,0,0))
        cap.release();sheet.save(OUT/(word+'_review.jpg'))
        row['shared_visual_crop']=crop
    (OUT/'media_evidence.json').write_text(json.dumps(dict(capture_reproduced=True,review=data['rows'],hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*_review.jpg')}),indent=2))
    print('Verified all captured joints; encoded10 videos and5 source-comparison contact sheets',flush=True)

if __name__=='__main__':main()

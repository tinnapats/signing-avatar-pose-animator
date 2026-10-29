"""Contact sheets from actual browser screenshots, selected by wrist discrepancy."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;OUT=HERE/'scale_anchor200_20261003'
p=json.loads((OUT/'protocol.json').read_text());g=json.loads((OUT/'geometry.json').read_text())['rows']
rank=sorted([r for r in g if r['variant']=='A' and r['label'] in p['review'] and r['wrist_to_pose_wrist_shoulders'] is not None],key=lambda r:r['wrist_to_pose_wrist_shoulders'],reverse=True)[:3]
records=[]
for r in rank:
    label=r['label'];i=p['labels'].index(label);a=np.load(OUT/'arrays'/f'{i:03d}.npz');n=len(a['A']);indices=[0,n//4,n//2,3*n//4,n-1]
    def gap(t):
        values=[]
        for o in [33,54]:
            if np.any(a['A'][t,o]):values.append(float(np.linalg.norm(a['A'][t,o]-a['C'][t,o])))
        return max(values,default=-1)
    t=max(indices,key=gap);sheet=Image.new('RGB',(900,960),'white');draw=ImageDraw.Draw(sheet)
    for j,v in enumerate('ABCD'):
        src=OUT/'renderer/frames'/f'{label}__no_hold_no_smoothing__{v}'/f'{t:04d}.png'
        im=Image.open(src).convert('RGB');im.thumbnail((450,450));x=(j%2)*450;y=(j//2)*480
        sheet.paste(im,(x,y+30));draw.text((x+8,y+8),f'{v} | {label} | frame {t}',fill='black')
    filename=label.replace(' ','_')+'_comparison.png';sheet.save(OUT/filename)
    records.append(dict(label=label,frame=t,path=filename,selection='Top3 word-mean wrist mismatch among fixed review20; highest visible mismatch among five saved timepoints'))
(OUT/'image_selection.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print(json.dumps(records,indent=2))

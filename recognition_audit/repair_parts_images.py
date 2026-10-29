"""Arrange actual Canvas captures, no synthesized poses."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
from pathlib import Path
import os
import json
from PIL import Image, ImageDraw
OUT=Path(os.environ.get('PARTS_OUTPUT',str(Path(__file__).resolve().parent/'repair_parts200')))
VS=_local_json_loads(os.environ.get('PARTS_VARIANTS','["baseline","no_short_fill","no_hold","no_repair"]'))
for label in ['never','parents','hurry']:
    folders=[OUT/'renderer/frames'/f'{label}__{v}' for v in VS]
    files=[sorted(f.glob('*.png')) for f in folders]
    assert len(set(len(f) for f in files))==1
    composites=[]
    for i in range(len(files[0])):
        panel=Image.new('RGB',(256*len(VS),284),'white');d=ImageDraw.Draw(panel)
        for j,v in enumerate(VS):
            with Image.open(files[j][i]) as im:panel.paste(im.convert('RGB').resize((256,256)),(j*256,28))
            d.text((j*256+4,4),v,fill='black')
        composites.append(panel)
    # GIF duration is quantized to 10 ms; repeating 30/30/40 ms gives 30 FPS on average.
    composites[0].save(OUT/f'{label}_comparison.gif',save_all=True,append_images=composites[1:],duration=[30,30,40]*(len(composites)//3)+[30,30,40][:len(composites)%3],loop=0)
    chosen=sorted(set([0,len(composites)//4,len(composites)//2,3*len(composites)//4,len(composites)-1]))
    sheet=Image.new('RGB',(256*len(VS),304*len(chosen)),'white');d=ImageDraw.Draw(sheet)
    for j,i in enumerate(chosen):
        sheet.paste(composites[i],(0,j*304));d.text((4,j*304+285),f'{label}: sampled frame {i}, time {i/30:.2f}s',fill='black')
    sheet.save(OUT/f'{label}_filmstrip.png')
    print(label,len(composites),flush=True)

"""Contact sheets from actual Canvas screenshots, not generated poses."""
from pathlib import Path
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent/'combined_winners200'
items=[('baseline','S1A1','Baseline'),('no_hold','S1A1','No hold'),('no_hold_no_smoothing','S1A1','No hold + no Gaussian'),('no_hold_no_smoothing','S50A1','Combination + S50A1')]
for label in ['never','parents','hurry']:
    folders=[OUT/'renderer/frames'/f'{label}__{v}__{s}' for v,s,_ in items]
    files=[sorted(f.glob('*.png')) for f in folders];assert all(len(f)==5 for f in files)
    sheet=Image.new('RGB',(1280,365*5),'white');d=ImageDraw.Draw(sheet)
    for row in range(5):
        for col,(_,_,title) in enumerate(items):
            with Image.open(files[col][row]) as im:sheet.paste(im.convert('RGB').resize((320,320)),(col*320,row*365+24))
            d.text((col*320+4,row*365+4),title,fill='black')
        d.text((4,row*365+345),f'{label}: frame {int(files[0][row].stem)}, {int(files[0][row].stem)/30:.2f}s',fill='black')
    sheet.save(OUT/f'{label}_comparison.png')
    print(label,flush=True)

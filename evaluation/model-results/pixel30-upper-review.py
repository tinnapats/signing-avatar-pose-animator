from pathlib import Path
import json,numpy as np,cv2
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'upper_crop_20261008';cid='001';i=45
meta=json.loads((O/(cid+'.json')).read_text());scale=meta['scale'];origin=np.array(meta['origin']);mat=np.array([[scale,0,-origin[0]*scale],[0,scale,-origin[1]*scale]],np.float32)
im=cv2.imread(str(B/'frames'/cid/f'{i:04d}.png'),cv2.IMREAD_UNCHANGED);a=im[:,:,3:4]/255;im=np.uint8(np.rint(im[:,:,:3]*a+255*(1-a)));crop=cv2.warpAffine(im,mat,(808,808),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255));points=np.load(O/(cid+'.npz'))['crop_raw'][i]
for off in [33,54]:
 if np.any(points[off:off+21]):
  for p in points[off:off+21]:cv2.circle(crop,tuple(np.rint((p-origin)*scale).astype(int)),4,(190,0,190),-1)
panels=[]
for image,label in [(im,'Original Canvas: hello frame45'),(crop,f'Upper crop x{scale:.2f}; purple=detected')]:
 image=cv2.resize(image,(404,404));image=cv2.copyMakeBorder(image,40,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(image,label,(7,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1);panels.append(image)
cv2.imwrite(str(O/'review.png'),np.concatenate(panels,axis=1))

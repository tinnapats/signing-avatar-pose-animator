// Reuse the production-mode audit build, replacing only the capture routine.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let source=await fs.readFile(path.join(here,'render_hands.mjs'),'utf8');
source=source.replace('const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));',`const root=${JSON.stringify(path.dirname(here))};`);
const start=source.indexOf('const hook=`'),end=source.indexOf('await build(',start);
if(start<0||end<0)throw Error('Missing capture insertion point');
const hook=`window.runFidelityCase=async(payload,indices)=>{
 stopPlayback();loadSequencePayload(payload);
 if(handRigReady || !showHandOverlays)throw Error('Unexpected renderer');
 const images=[];
 for(const index of indices){renderFrame(index);canvasScope.view.update();
  const canvas=document.querySelector('canvas.illustration-canvas');
  if(canvas.width<=0 || canvas.height<=0)throw Error('Empty canvas');
  images.push(canvas.toDataURL('image/png').split(',')[1]);
 }
 const canvas=document.querySelector('canvas.illustration-canvas');
 return {images,width:canvas.width,height:canvas.height};
};`;
source=source.slice(0,start)+'const hook='+JSON.stringify(hook)+';\n'+source.slice(end);
const loopStart=source.indexOf(' for(const test of cases){'),loopEnd=source.indexOf(' if(errors.length)',loopStart);
if(loopStart<0||loopEnd<0)throw Error('Missing loop');
source=source.slice(0,loopStart)+` for(let c=0;c<cases.length;c++){
 const test=cases[c],fps=test.payload.meta.fps,n=test.payload.frames.length;
 const indices=Array.from({length:Math.floor((n-1)/fps*25)+1},(_,i)=>Math.min(n-1,Math.round(i*fps/25)));
 const capture=await page.evaluate(({payload,indices})=>window.runFidelityCase(payload,indices),{payload:test.payload,indices});
 const images=capture.images;
 const id=String(c).padStart(3,'0');const folder=path.join(auditOut,'frames',id);await fs.mkdir(folder,{recursive:true});
 for(let i=0;i<images.length;i++)await fs.writeFile(path.join(folder,String(i).padStart(4,'0')+'.png'),Buffer.from(images[i],'base64'));
 results.push({id,phrase:test.phrase,fps:25,sourceFps:fps,indices,frames:images.length,width:capture.width,height:capture.height});console.log('canvas captured',id,test.phrase,images.length,capture.width,capture.height);
 }
`+source.slice(loopEnd);
source=source.replace("path.join(auditOut,'rendered.json')","path.join(auditOut,'capture_manifest.json')");
process.env.AUDIT_OUTPUT=path.join(here,'i3d_avatar');
process.env.AUDIT_INPUTS=path.join(here,'inputs.json');
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

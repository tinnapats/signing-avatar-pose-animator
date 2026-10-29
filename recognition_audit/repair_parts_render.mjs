// Instrument the established renderer audit only; production JS stays unchanged.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let source=await fs.readFile(path.join(here,'render_hands.mjs'),'utf8');
function replace(a,b){if(source.split(a).length!==2)throw Error('Nonunique hook: '+a);source=source.replace(a,b);}
replace('const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));',`const root=${JSON.stringify(path.dirname(here))};`);
replace('results.push({hands,zoom:WAIST_CAMERA_ZOOM});',`canvasScope.view.update();
  let png=null;
  if(window.auditSaveAll || window.auditSnapshots.includes(index)){
   const original=document.querySelector('canvas.illustration-canvas');
   const copy=document.createElement('canvas');copy.width=original.width;copy.height=original.height;
   const ctx=copy.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,copy.width,copy.height);ctx.drawImage(original,0,0);
   png=copy.toDataURL('image/png').split(',')[1];
  }
  results.push({hands,zoom:WAIST_CAMERA_ZOOM,png,index});`);
replace("const samples=await page.evaluate(({payload,frameIndices})=>window.runFidelityCase(payload,frameIndices),{payload:test.payload,frameIndices:test.frameIndices});",`const samples=await page.evaluate(({payload,frameIndices,label})=>{
   window.auditSaveAll=['never','parents','hurry'].includes(label);
   const candidates=[];
   for(let i=2;i<payload.frames.length;i+=2){
    const a=payload.frames[i-2],b=payload.frames[i];
    if(['left','right'].some(s=>!!a.hands[s].observed!==!!b.hands[s].observed))candidates.push(i);
   }
   window.auditSnapshots=[0,frameIndices[Math.floor(frameIndices.length/2)],frameIndices.at(-1),...candidates];
   return window.runFidelityCase(payload,frameIndices);
  },{payload:test.payload,frameIndices:test.frameIndices,label:test.label});
  const folder=path.join(auditOut,'frames',test.phrase);await fs.mkdir(folder,{recursive:true});
  for(const sample of samples){if(sample.png)await fs.writeFile(path.join(folder,String(sample.index).padStart(4,'0')+'.png'),Buffer.from(sample.png,'base64'));delete sample.png;}`);
process.env.AUDIT_OUTPUT ||= path.join(here,'repair_parts200','renderer');
process.env.AUDIT_INPUTS ||= path.join(here,'repair_parts200','review_inputs.json');
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

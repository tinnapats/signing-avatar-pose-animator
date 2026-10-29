// Capture full motion of five diagnostic cases with existing two frozen variants.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let source=await fs.readFile(path.join(here,'step1b_render.mjs'),'utf8');
source=source.replace('const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));',`const root=${JSON.stringify(path.dirname(here))};`);
const needle='results.push({hands,zoom:WAIST_CAMERA_ZOOM});';
if(source.split(needle).length!==2)throw Error('Capture hook mismatch');
source=source.replace(needle,`canvasScope.view.update();
 const original=document.querySelector('canvas.illustration-canvas');
 const exportCanvas=document.createElement('canvas');exportCanvas.width=original.width;exportCanvas.height=original.height;
 const ctx=exportCanvas.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,exportCanvas.width,exportCanvas.height);ctx.drawImage(original,0,0);
 results.push({hands,zoom:WAIST_CAMERA_ZOOM,png:exportCanvas.toDataURL('image/png').split(',')[1]});`);
const insert='results.push({variant:options.id,phrase:test.phrase,samples});';
if(source.split(insert).length!==2)throw Error('Output hook mismatch');
source=source.replace(insert,`const folder=path.join(out,'frames',options.id,test.phrase);await fs.mkdir(folder,{recursive:true});
 for(let i=0;i<samples.length;i++){await fs.writeFile(path.join(folder,String(i).padStart(4,'0')+'.png'),Buffer.from(samples[i].png,'base64'));delete samples[i].png;}
 ${insert}`);
process.env.AUDIT_OUTPUT=path.join(here,'method_review');process.env.AUDIT_INPUTS=path.join(here,'method_review/inputs.json');
process.env.AUDIT_VARIANTS=JSON.stringify([{id:'S1A1',scale:1,anchor:1},{id:'S50A1',scale:.5,anchor:1}]);
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

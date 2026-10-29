import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let code=await fs.readFile(path.join(here,'step1b_render.mjs'),'utf8');
function replace(a,b){if(code.split(a).length!==2)throw Error('Nonunique '+a);code=code.replace(a,b);}
replace('const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));',`const root=${JSON.stringify(path.dirname(here))};`);
replace('results.push({hands,zoom:WAIST_CAMERA_ZOOM});',`const arms={}; for(const side of ['left','right']) for(const joint of ['Shoulder','Elbow','Wrist']) {const name=side+joint;const p=illustration.skeleton.parts[name].position;arms[name]=[p.x,p.y];} results.push({hands,arms,zoom:WAIST_CAMERA_ZOOM});`);
replace('for(const options of variants) for(const test of cases){',`for(const descriptor of cases){const test=JSON.parse(await fs.readFile(descriptor.file,'utf8'));const options={id:test.method,scale:0,anchor:1};`);
replace("if(['hello','love','abdomen','hard','loan'].includes(test.phrase)) await page.screenshot({path:path.join(out,options.id+'_'+test.phrase.replaceAll(' ','_')+'.png')});",`for(const fraction of [0,.25,.5,.75,1]) {await page.evaluate(index=>window.showAuditFrame(index),test.frameIndices[Math.floor((test.frameIndices.length-1)*fraction)]);await page.locator('canvas.illustration-canvas').screenshot({path:path.join(out,test.phrase.replaceAll(' ','_')+'_'+Math.round(fraction*100)+'.png')});}`);
process.env.AUDIT_OUTPUT=path.join(here,'bounded_canvas_20261005/renderer');
process.env.AUDIT_INPUTS=path.join(here,'bounded_canvas_20261005/case_index.json');
try{await fs.access(path.join(process.env.AUDIT_OUTPUT,'rendered.json'));throw Error('Capture exists');}catch(e){if(e.code!=='ENOENT')throw e;}
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

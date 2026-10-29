import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const review=process.argv.includes('--review');
let code=await fs.readFile(path.join(here,'step1b_render.mjs'),'utf8');
function replace(a,b){if(code.split(a).length!==2)throw Error('Nonunique '+a);code=code.replace(a,b);}
replace('const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));',`const root=${JSON.stringify(path.dirname(here))};`);
replace('stopPlayback(); loadSequencePayload(payload);','stopPlayback(); loadSequencePayload(payload); handOverlayScales={...payload.auditScales};');
replace('results.push({hands,zoom:WAIST_CAMERA_ZOOM});',`const arms={}; for(const side of ['left','right']) for(const joint of ['Shoulder','Elbow','Wrist']) {const name=side+joint; const p=illustration.skeleton.parts[name].position; arms[name]=[p.x,p.y];} results.push({hands,arms,zoom:WAIST_CAMERA_ZOOM});`);
replace('for(const options of variants) for(const test of cases){',`for(const descriptor of cases){ const test=JSON.parse(await fs.readFile(descriptor.file,'utf8')); const options={id:test.method,scale:1,anchor:test.method==='HAND_TO_BODY'?1:0};`);
replace("if(['hello','love','abdomen','hard','loan'].includes(test.phrase)) await page.screenshot({path:path.join(out,options.id+'_'+test.phrase.replaceAll(' ','_')+'.png')});",`if(['parents','chicken','remote control'].includes(test.label)){await page.locator('canvas.illustration-canvas').screenshot({path:path.join(out,test.phrase.replaceAll(' ','_')+'.png')});} if(results.length%20===0)await fs.writeFile(path.join(out,'progress.json'),JSON.stringify({completed:results.length}));`);
process.env.AUDIT_OUTPUT=path.join(here,'arm_hand_compare_20261003/renderer');
process.env.AUDIT_INPUTS=path.join(here,'arm_hand_compare_20261003/case_index.json');
if(review){
  process.env.AUDIT_OUTPUT=path.join(here,'arm_hand_compare_20261003/visual_review');
  replace('for(const descriptor of cases){',"for(const descriptor of cases.filter(c=>['parents','chicken','remote control'].includes(c.label))){");
  replace("if(results.length%20===0)",`for(const fraction of [0,.25,.5,.75,1]){const frame=test.frameIndices[Math.floor((test.frameIndices.length-1)*fraction)];await page.evaluate(index=>window.showAuditFrame(index),frame);await page.locator('canvas.illustration-canvas').screenshot({path:path.join(out,test.phrase.replaceAll(' ','_')+'_'+Math.round(fraction*100)+'.png')});} if(results.length%20===0)`);
}
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

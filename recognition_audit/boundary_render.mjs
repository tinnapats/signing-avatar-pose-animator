import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let code=await fs.readFile(path.join(here,'combined_render.mjs'),'utf8');
code=code.replace('const here=path.dirname(fileURLToPath(import.meta.url));',`const here=${JSON.stringify(here)};`);
const needle='// Save a few diagnostic actual Canvas frames, rather than synthetic diagrams.';
if(code.split(needle).length!==2)throw Error('Hook mismatch');
code=code.replace(needle,`
replace('window.auditOptions=options;', 'window.auditOptions=options; window.auditBodyRefs=new Map();');
replace('for(const options of variants) for(const test of cases){', "for(const options of variants) for(const descriptor of cases){ const test=JSON.parse(await fs.readFile(descriptor.file,'utf8')); ");
replace('results.push({hands,native,index,zoom:WAIST_CAMERA_ZOOM});', \`
 const geometry={};
 const ls=getPoseKeypoint(frame.pose,'leftShoulder').position,rs=getPoseKeypoint(frame.pose,'rightShoulder').position;
 for(const side of ['left','right']){
  const paths=canvasScope.project.activeLayer.children.filter(p=>(p.data.auditSide===side || p.data.auditNativeSide===side) && p.visible && p.opacity>0);
  if(!paths.length){geometry[side]=null;continue;}
  const bs=paths.map(p=>p.strokeBounds);const a=unproject(Math.min(...bs.map(b=>b.left)),Math.min(...bs.map(b=>b.top)));const z=unproject(Math.max(...bs.map(b=>b.right)),Math.max(...bs.map(b=>b.bottom)));
  const w=getPoseKeypoint(frame.pose,side+'Wrist').position;
  geometry[side]={bounds:[...a,...z],wrist:[w.x,w.y],shoulderWidth:Math.hypot(ls.x-rs.x,ls.y-rs.y)};
 }
 results.push({hands,native,index,geometry,zoom:WAIST_CAMERA_ZOOM});\`);
${needle}`);
const review=process.argv.includes('--review');
process.env.AUDIT_OUTPUT=path.join(here,'boundary_audit_20261003',review?'visual_review':'renderer');
process.env.AUDIT_INPUTS=path.join(here,'boundary_audit_20261003',review?'review_index.json':'case_index.json');
process.env.AUDIT_VARIANTS=JSON.stringify([{id:'S1A1',scale:1,anchor:1}]);
if(review)process.env.AUDIT_CAPTURE_SEQUENCE='1';
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

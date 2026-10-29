import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let code=await fs.readFile(path.join(here,'combined_render.mjs'),'utf8');
// Expand wrapper without changing original experiment or production sources.
code=code.replace('const here=path.dirname(fileURLToPath(import.meta.url));',`const here=${JSON.stringify(here)};`);
const marker="await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));";
if(code.split(marker).length!==2)throw Error('Wrapper changed');
code=code.replace(marker,`
replace('window.auditOptions=options;', 'window.auditOptions=options; window.auditBodyRefs=new Map();');
replace('results.push({hands,native,index,zoom:WAIST_CAMERA_ZOOM});', "const arms={};for(const side of ['left','right'])for(const joint of ['Shoulder','Elbow','Wrist']){const name=side+joint;const p=illustration.skeleton.parts[name].position;arms[name]=[p.x,p.y];}results.push({hands,native,arms,index,zoom:WAIST_CAMERA_ZOOM});");
replace('for(const options of variants) for(const test of cases){', "for(const descriptor of cases){const saved=path.join(out,'capture_'+path.basename(descriptor.file,'.json')+'.json');try{await fs.access(saved);continue;}catch(e){if(e.code!=='ENOENT')throw e;} const test=JSON.parse(await fs.readFile(descriptor.file,'utf8'));const options={id:test.method,scale:test.method==='BASE'?1:0,anchor:1};");
replace('results.push({variant:options.id,phrase:test.phrase,samples});console.log(options.id,test.phrase,samples.length);', "await fs.writeFile(saved,JSON.stringify({variant:options.id,phrase:test.phrase,samples}));console.log(options.id,test.phrase,samples.length);");
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
`);
process.env.AUDIT_OUTPUT=path.join(here,'frozen100_20261005/renderer');
process.env.AUDIT_INPUTS=path.join(here,'frozen100_20261005/case_index.json');
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

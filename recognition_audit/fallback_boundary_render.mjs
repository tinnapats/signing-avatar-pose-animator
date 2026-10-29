import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let wrapper=await fs.readFile(path.join(here,'boundary_render.mjs'),'utf8');
wrapper=wrapper.replace('const here=path.dirname(fileURLToPath(import.meta.url));',`const here=${JSON.stringify(here)};`);
const adjustment=await fs.readFile(path.join(here,'fallback_boundary_adjust.js'),'utf8');
const needle='await import(\'data:text/javascript;base64,\'+Buffer.from(code).toString(\'base64\'));';
if(wrapper.split(needle).length!==2)throw Error('Wrapper hook mismatch');
const injection=`
replace(" replace('void pollMicReadiness();','/* audit: no microphone model startup */');", " replace('void pollMicReadiness();','/* audit: no microphone model startup */');\\n replace('applyWaistCameraFraming(frame.pose);', " + ${JSON.stringify(JSON.stringify(adjustment+'\napplyWaistCameraFraming(frame.pose);'))} + ");");
replace('results.push({hands,native,index,geometry,zoom:WAIST_CAMERA_ZOOM});','results.push({hands,native,index,geometry,body,ops:window.auditFallbackOps,zoom:WAIST_CAMERA_ZOOM});');
replace("path.join(out,'frames',test.phrase)","path.join(out,'frames',test.phrase+'__'+options.id)");
`;
wrapper=wrapper.replace(needle,`
code=code.replace('// Save a few diagnostic actual Canvas frames, rather than synthetic diagrams.',${JSON.stringify(injection)}+'\\n// Save a few diagnostic actual Canvas frames, rather than synthetic diagrams.');
process.env.AUDIT_OUTPUT=path.join(here,'fallback_boundary_20261003',review?'visual_review':'renderer');
process.env.AUDIT_INPUTS=path.join(here,'fallback_boundary_20261003',review?'review_index.json':'case_index.json');
process.env.AUDIT_VARIANTS=JSON.stringify([{id:'OFF',scale:1,anchor:1,fallbackAdjust:false},{id:'ON',scale:1,anchor:1,fallbackAdjust:true}]);
${needle}`);
await import('data:text/javascript;base64,'+Buffer.from(wrapper).toString('base64'));

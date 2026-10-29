import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let code=await fs.readFile(path.join(here,'combined_render.mjs'),'utf8');
code=code.replace('const here=path.dirname(fileURLToPath(import.meta.url));',`const here=${JSON.stringify(here)};`);
const needle='// Save a few diagnostic actual Canvas frames, rather than synthetic diagrams.';
// Run all variants for one clip together, retaining the body-reference assertion.
code=code.replace(needle,`replace('for(const options of variants) for(const test of cases){', 'for(const test of cases) for(const options of variants){');
replace('window.auditOptions=options;', 'window.auditOptions=options; if(options.id === "A") window.auditBodyRefs=new Map();');
${needle}`);
code=code.replace("if(['never','parents','hurry'].includes(test.label)){","if(true){");
process.env.AUDIT_OUTPUT=path.join(here,'scale_anchor200_20261003/renderer');
process.env.AUDIT_INPUTS=path.join(here,'scale_anchor200_20261003/review_inputs.json');
process.env.AUDIT_VARIANTS=JSON.stringify([{id:'A',scale:0,anchor:0},{id:'B',scale:1,anchor:0},{id:'C',scale:0,anchor:1},{id:'D',scale:1,anchor:1}]);
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

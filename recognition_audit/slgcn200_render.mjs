// Same production renderer instrumented in a temporary bundle; one fixed scale.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let code=await fs.readFile(path.join(here,'combined_render.mjs'),'utf8');
code=code.replace('const here=path.dirname(fileURLToPath(import.meta.url));',`const here=${JSON.stringify(here)};`);
// Bound memory: only one scale, no cross-case experiment needs this reference.
const needle="// Save a few diagnostic actual Canvas frames, rather than synthetic diagrams.";
if(code.split(needle).length!==2)throw Error('Hook changed');
code=code.replace(needle,`replace('window.auditOptions=options;', 'window.auditOptions=options; window.auditBodyRefs=new Map();');\n${needle}`);
process.env.AUDIT_OUTPUT=path.join(here,'slgcn200_20261002/renderer');
process.env.AUDIT_INPUTS=path.join(here,'slgcn200_20261002/inputs.json');
process.env.AUDIT_VARIANTS=JSON.stringify([{id:'S1A1',scale:1,anchor:1}]);
try{await fs.access(path.join(process.env.AUDIT_OUTPUT,'rendered.json'));throw Error('Refuse to overwrite completed capture');}catch(e){if(e.code!=='ENOENT')throw e;}
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

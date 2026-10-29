// Frozen previous chain cap; S1A1; reuse diagnostic12 without production edits.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const out=path.join(here,'transition_chain_20261002');
await fs.mkdir(out); // Refuse to overwrite prior evidence.
const all=JSON.parse(await fs.readFile(path.join(here,'bridge_transition_20261002/review_inputs.json'),'utf8'));
const cases=all.filter(c=>c.variant==='control');
if(cases.length!==12)throw Error('Expected diagnostic12');
await fs.writeFile(path.join(out,'inputs.json'),JSON.stringify(cases));
await fs.writeFile(path.join(out,'protocol.json'),JSON.stringify({labels:cases.map(c=>c.label),
  note:'Previously fixed chain cap, no new tuning. Diagnostic12 only. Same input, S1A1. Cap native SVG only around body wrist; never changes detailed joints, no inferred native joints. Offline whole-clip reference.',
  variants:[{id:'control',scale:1,anchor:1,cap:false},{id:'chain',scale:1,anchor:1,cap:true,reference:'chain'}]},null,2));
let source=await fs.readFile(path.join(here,'fallback_trial_render.mjs'),'utf8');
source=source.replace("from './fallback_cap.mjs'",`from '${pathToFileURL(path.join(here,'fallback_cap.mjs')).href}'`);
source=source.replace('const here=path.dirname(fileURLToPath(import.meta.url));',`const here=${JSON.stringify(here)};`);
source=source.replace("process.env.AUDIT_INPUTS=path.join(here,'method_review/inputs.json');",`process.env.AUDIT_INPUTS=${JSON.stringify(path.join(out,'inputs.json'))};`);
process.env.FALLBACK_OUTPUT=out;
process.env.FALLBACK_VARIANTS=JSON.stringify([{id:'control',scale:1,anchor:1,cap:false},{id:'chain',scale:1,anchor:1,cap:true,reference:'chain'}]);
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

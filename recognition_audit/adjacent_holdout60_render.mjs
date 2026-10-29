// Resolve the frozen audit wrappers in memory; never edit production sources.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const out=path.join(here,'adjacent_holdout60_20261004');
let code=await fs.readFile(path.join(here,'adjacent_reference_render.mjs'),'utf8');
code=code.replace('const here=path.dirname(fileURLToPath(import.meta.url));',`const here=${JSON.stringify(here)};`);
const terminal=/await import\('data:text\/javascript;base64,'\+Buffer.from\((source|wrapper|code)\).toString\('base64'\)\);/;
for(let depth=0;terminal.test(code);depth++){
 if(depth>8)throw Error('Unexpected wrapper depth');
 const variable=code.match(terminal)[1];
 const stage=code.replace(terminal,`export const generated=${variable};`);
 code=(await import('data:text/javascript;base64,'+Buffer.from(stage).toString('base64'))).generated;
}
function replace(a,b){if(code.split(a).length!==2)throw Error('Nonunique instrumentation hook '+a);code=code.replace(a,b);}
replace('  renderFrame(index);\n  const frame=', '  const renderStarted=performance.now(); renderFrame(index); const renderMs=performance.now()-renderStarted;\n  const frame=');
replace('results.push({hands,native,index,geometry,body,ops:window.auditFallbackOps,zoom:WAIST_CAMERA_ZOOM});','results.push({hands,native,index,geometry,body,ops:window.auditFallbackOps,zoom:WAIST_CAMERA_ZOOM,renderMs});');
replace('for(const options of variants) for(const descriptor of cases){',`// Warm up all methods, excluded from timing results.
 for(const options of variants){const warm=JSON.parse(await fs.readFile(cases[0].file,'utf8'));await page.evaluate(({payload,frameIndices,options})=>window.runFidelityCase(payload,frameIndices,options),{payload:warm.payload,frameIndices:warm.frameIndices.slice(0,20),options});}
 for(const [caseNumber,descriptor] of cases.entries()) for(const options of [...variants.slice(caseNumber%3),...variants.slice(0,caseNumber%3)]){`);
const review=process.argv.includes('--review'), canary=process.argv.includes('--canary');
process.env.AUDIT_OUTPUT=path.join(out,canary?'canary':review?'visual_review':'renderer');
process.env.AUDIT_INPUTS=path.join(out,canary?'canary_index.json':review?'review_index.json':'case_index.json');
if(review)process.env.AUDIT_CAPTURE_SEQUENCE='1';
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

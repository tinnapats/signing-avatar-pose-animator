import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let code=await fs.readFile(path.join(here,'bounded_canvas_render.mjs'),'utf8');
code=code.replace('const here=path.dirname(fileURLToPath(import.meta.url));',`const here=${JSON.stringify(here)};`);
code=code.replaceAll('bounded_canvas_20261005','continuous_review_20261005');
code=code.replace('scale:0,anchor:1','scale:test.method===\'BASE\'?1:0,anchor:1');
const marker="await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));";
if(code.split(marker).length!==2)throw Error('Wrapper changed');
code=code.replace(marker,`
const needle="for(let i=0;i<test.frameIndices.length;i++){";
if(code.split(needle).length!==2)throw Error('Frame loop changed');
code=code.replace("await page.locator('canvas.illustration-canvas').screenshot({path:path.join(folder,String(i).padStart(4,'0')+'.png')});", "const data=await page.locator('canvas.illustration-canvas').evaluate(c=>c.toDataURL('image/png'));await fs.writeFile(path.join(folder,String(i).padStart(4,'0')+'.png'),Buffer.from(data.split(',')[1],'base64'));");
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
`);
process.env.AUDIT_CAPTURE_SEQUENCE='1';
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

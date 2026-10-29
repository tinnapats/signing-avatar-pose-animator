import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dir=path.dirname(fileURLToPath(import.meta.url));
const seed=JSON.parse(await fs.readFile(path.join(dir,'independent-check-20261012','annotations.json'),'utf8'));
if(seed.items.some(x=>Object.keys(x.joints).length!==46||Object.values(x.joints).some(j=>j.x!==null||j.y!==null)))throw Error('Reference prefilled or joint count wrong');
let code=await fs.readFile(path.join(dir,'test-annotator.mjs'),'utf8');
code=code.replaceAll('body-reference-v1-20261009','independent-check-20261012').replaceAll('path.dirname(fileURLToPath(import.meta.url))',JSON.stringify(dir)).replaceAll(',24)',','+seed.items.length+')').replace('24 image navigation',seed.items.length+' image navigation');
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

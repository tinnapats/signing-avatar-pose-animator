import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let source=await fs.readFile(path.join(here,'slgcn200_render.mjs'),'utf8');
source=source.replace('const here=path.dirname(fileURLToPath(import.meta.url));',`const here=${JSON.stringify(here)};`);
source=source.replaceAll('slgcn200_20261002/renderer','short_gap200_20261005/renderer').replaceAll('slgcn200_20261002/inputs.json','short_gap200_20261005/render_inputs.json');
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

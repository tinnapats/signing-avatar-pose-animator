import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath,pathToFileURL} from 'node:url';
const dir=path.dirname(fileURLToPath(import.meta.url)),out=path.join(dir,'inverse-animation-prf-20261013');
const source=path.join(dir,'oracle200-20261011/results.json'),bytes=await fs.readFile(source),data=JSON.parse(bytes),rows=data.predictions.both_oracle;
const labels=[...new Set(rows.map(r=>r.label))];assert.equal(rows.length,200);assert.equal(labels.length,200);
let p=0,r=0,f=0;
for(const l of labels){const tp=rows.filter(x=>x.label===l&&x.prediction===l).length,fp=rows.filter(x=>x.label!==l&&x.prediction===l).length,fn=rows.filter(x=>x.label===l&&x.prediction!==l).length;const pp=tp/(tp+fp)||0,rr=tp/(tp+fn)||0;p+=pp;r+=rr;f+=pp+rr?2*pp*rr/(pp+rr):0;}
const s={p:p/200,r:r/200,f:f/200,correct:rows.filter(x=>x.label===x.prediction).length};
for(const[k,v]of [['p','precision'],['r','recall'],['f','f1']])assert(Math.abs(s[k]*100-data.summary.all200.both_oracle[v])<1e-8);
const text=(x,y,t,size=28,attr='')=>`<text x="${x}" y="${y}" font-size="${size}" ${attr}>${t}</text>`;const center=(x,y,t,size=28)=>text(x,y,t,size,'text-anchor="middle"');
let q=center(800,62,'Precision–Recall–F1: Renderer-assisted recognition',38)+center(800,108,'ST-GCN | 200 clips | Reference-assisted diagnostic',27);
const x0=145,y0=175,w=1320,h=460;
for(let i=0;i<=6;i++){const y=y0+h-i*h/6;q+=`<line x1="${x0}" x2="${x0+w}" y1="${y}" y2="${y}" stroke="#d9d9d9" stroke-width="2"/>`+text(x0-25,y+8,(i*.2).toFixed(1),25,'text-anchor="end"');}
q+=text(52,405,'Macro score',26,'transform="rotate(-90 52 405)" text-anchor="middle"');
const colors=['#5B9BD5','#ED7D31','#A5A5A5'];['p','r','f'].forEach((k,i)=>{const x=800+(i-1.5)*165,hh=s[k]/1.2*h;q+=`<rect x="${x}" y="${y0+h-hh}" width="150" height="${hh}" fill="${colors[i]}"/>`+center(x+75,y0+h-hh-16,s[k].toFixed(4),30);});
q+=center(800,685,'ST-GCN — Renderer-assisted',30)+center(800,730,`Accuracy: ${s.correct}/200 = ${(s.correct/2).toFixed(2)}%`,28);
['Precision','Recall','F1-score'].forEach((t,i)=>{const x=475+i*235;q+=`<rect x="${x}" y="784" width="25" height="25" fill="${colors[i]}"/>`+text(x+36,807,t,28);});
q+=center(800,860,'RTMW image body + Renderer hand coordinates and reference-presence mask',24)+center(800,905,'NOT pixel-only extraction. NOT visual similarity or human ASL comprehension.',24)+center(800,946,'Macro over 200 true classes | 2,000 output classes | Development set',23);
await fs.writeFile(path.join(out,'renderer-assisted-prf.svg'),`<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="990"><style>text{font-family:Calibri,Arial,sans-serif;fill:#222}</style><rect width="1600" height="990" fill="white"/>${q}</svg>`);
await fs.writeFile(path.join(out,'renderer-assisted-verified.json'),JSON.stringify({source,sha256:crypto.createHash('sha256').update(bytes).digest('hex'),variant:'both_oracle',scores:s,scope:'Reference-assisted diagnostic, not pixel-only.'},null,2));
const {chromium}=await import('file:///C:/Users/newtu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
try{const page=await browser.newPage({viewport:{width:1600,height:990},deviceScaleFactor:2});await page.goto(pathToFileURL(path.join(out,'renderer-assisted-prf.svg')).href);await page.evaluate(()=>document.fonts.ready);await page.locator('svg').screenshot({path:path.join(out,'renderer-assisted-prf.png')});}finally{await browser.close();}
console.log(JSON.stringify(s));

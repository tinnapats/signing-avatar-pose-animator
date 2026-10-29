import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath,pathToFileURL} from 'node:url';
const dir=path.dirname(fileURLToPath(import.meta.url)),out=path.join(dir,'inverse-animation-prf-20261013');await fs.mkdir(out,{recursive:true});
const source=path.join(dir,'pixel-only-plan-20261012/results.json'),bytes=await fs.readFile(source),data=JSON.parse(bytes);
const variants=[['baseline','RTMW baseline'],['support_gate','RTMW + pixel-support filter']];
const scores=variants.map(([v,name])=>{const rows=data.predictions[v],labels=[...new Set(rows.map(r=>r.label))];assert.equal(rows.length,200);assert.equal(labels.length,200);let p=0,r=0,f=0;for(const l of labels){const tp=rows.filter(x=>x.label===l&&x.prediction===l).length,fp=rows.filter(x=>x.label!==l&&x.prediction===l).length,fn=rows.filter(x=>x.label===l&&x.prediction!==l).length;const pp=tp/(tp+fp)||0,rr=tp/(tp+fn)||0;p+=pp;r+=rr;f+=pp+rr?2*pp*rr/(pp+rr):0;}const result={name,variant:v,p:p/200,r:r/200,f:f/200,correct:rows.filter(x=>x.label===x.prediction).length};for(const [k,s]of [['p','precision'],['r','recall'],['f','f1']])assert(Math.abs(result[k]*100-data.summary.all200[v][s])<1e-8);return result;});
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');const text=(x,y,s,size=24,attr='')=>`<text x="${x}" y="${y}" font-size="${size}" ${attr}>${esc(s)}</text>`;const center=(x,y,s,size=24)=>text(x,y,s,size,'text-anchor="middle"');
let q=center(900,65,'Precision–Recall–F1: Animation images to keypoints',38)+center(900,110,'Same ST-GCN recognizer | 200 clips | Pixel-only input',28);
const x0=145,y0=185,w=1500,h=485;
for(let i=0;i<=6;i++){const y=y0+h-i*h/6;q+=`<line x1="${x0}" x2="${x0+w}" y1="${y}" y2="${y}" stroke="#d9d9d9" stroke-width="2"/>`+text(x0-25,y+8,(i*.2).toFixed(1),25,'text-anchor="end"');}
q+=text(55,430,'Macro score',27,'transform="rotate(-90 55 430)" text-anchor="middle"');
const colors=['#5B9BD5','#ED7D31','#A5A5A5'];scores.forEach((s,i)=>{const cx=x0+w*(i+.5)/2;['p','r','f'].forEach((k,j)=>{const x=cx+(j-1.5)*132,hh=s[k]/1.2*h;q+=`<rect x="${x}" y="${y0+h-hh}" width="120" height="${hh}" fill="${colors[j]}"/>`+center(x+60,y0+h-hh-14,s[k].toFixed(4),26);});q+=center(cx,720,s.name,28)+center(cx,765,`Accuracy: ${s.correct}/200 = ${(s.correct/2).toFixed(2)}%`,26);});
['Precision','Recall','F1-score'].forEach((s,i)=>{const x=580+i*225;q+=`<rect x="${x}" y="825" width="25" height="25" fill="${colors[i]}"/>`+text(x+36,847,s,27);});
q+=center(900,905,'No Renderer reference coordinates or reference masks used at inference.',24)+center(900,947,'Macro over 200 true classes; predictions unrestricted across 2,000 classes.',22)+center(900,987,'Exploratory development set; recognition accuracy is not visual similarity or ASL comprehension.',21);
const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1030" viewBox="0 0 1800 1030"><style>text{font-family:Calibri,Arial,sans-serif;fill:#222}</style><rect width="1800" height="1030" fill="white"/>${q}</svg>`;
await fs.writeFile(path.join(out,'inverse-animation-prf.svg'),svg);await fs.writeFile(path.join(out,'verified-scores.json'),JSON.stringify({source,sha256:crypto.createHash('sha256').update(bytes).digest('hex'),scores,note:'Both methods use ST-GCN. Not ST-GCN vs SL-GCN; no renderer oracle.'},null,2));
const {chromium}=await import('file:///C:/Users/newtu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
try{const page=await browser.newPage({viewport:{width:1800,height:1030},deviceScaleFactor:2});await page.goto(pathToFileURL(path.join(out,'inverse-animation-prf.svg')).href);await page.evaluate(()=>document.fonts.ready);await page.locator('svg').screenshot({path:path.join(out,'inverse-animation-prf.png')});}finally{await browser.close();}
console.log(JSON.stringify(scores));

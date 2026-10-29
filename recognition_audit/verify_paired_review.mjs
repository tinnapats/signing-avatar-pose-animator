import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const {chromium}=await import('file:///C:/Users/newtu/signing-avatar-github-newton-upload/frontend-tooling/node_modules/playwright/index.mjs');
const out=path.join(path.dirname(fileURLToPath(import.meta.url)),'paired_video');
const browser=await chromium.launch({headless:true,executablePath:'C:/Users/newtu/signing-avatar-github-newton-upload/local_runtime/chrome/chrome.exe'});
try{
 const page=await browser.newPage({viewport:{width:1200,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.join(out,'compare.html')).href);
 try{await page.waitForFunction(()=>window.reviewCases && [...document.querySelectorAll('video')].every(v=>v.readyState>=1),null,{timeout:10000});}catch(e){console.log(await page.evaluate(()=>({cases:window.reviewCases,videos:[...document.querySelectorAll('video')].map(v=>({src:v.src,state:v.readyState,error:v.error?.message}))})));throw e;}
 const counts=await page.evaluate(()=>window.reviewCases);
 for(const c of counts)for(let i=0;i<c.frames;i++)await fs.access(path.join(out,'frames',c.word,String(i).padStart(4,'0')+'.png'));
 await page.evaluate(()=>{for(const v of document.querySelectorAll('video'))v.currentTime=v.duration/2;});
 await page.waitForFunction(()=>[...document.querySelectorAll('img')].every(i=>i.complete&&i.naturalWidth>0)&&[...document.querySelectorAll('video')].every(v=>!v.seeking));
 await page.screenshot({path:path.join(out,'review_preview.png'),fullPage:true});
 if(errors.length)throw new Error(errors.join('\n'));
 console.log(JSON.stringify({clips:counts.length,frames:counts.reduce((n,c)=>n+c.frames,0),videos_loaded:true,images_loaded:true,page_errors:errors}));
}finally{await browser.close();}

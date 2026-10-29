import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const {chromium}=await import(pathToFileURL('C:/Users/newtu/signing-avatar-github-newton-upload/frontend-tooling/node_modules/playwright/index.mjs'));
const out=path.join(path.dirname(fileURLToPath(import.meta.url)),'adjacent_holdout60_20261004/visual_review');
const browser=await chromium.launch({headless:true,executablePath:'C:/Users/newtu/signing-avatar-github-newton-upload/local_runtime/chrome/chrome.exe'});
try{
 const page=await browser.newPage({viewport:{width:1400,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.join(out,'compare.html')).href);
 await page.waitForFunction(()=>document.querySelector('#loaded').textContent==='โหลดภาพ 234/234 ภาพ');
 for(let i=0;i<3;i++){
  const section=page.locator('section').nth(i);
  await section.screenshot({path:path.join(out,`comparison_${i}_before.png`)});
  await section.locator('.next').click();
  if(!(await section.locator('.status').textContent()).includes('หลัง'))throw Error('Next-frame control failed');
  await page.waitForFunction(()=>[...document.images].every(im=>im.complete && im.naturalWidth>0));
  await section.screenshot({path:path.join(out,`comparison_${i}_after.png`)});
  await section.locator('.prev').click();
  if(!(await section.locator('.status').textContent()).includes('ก่อน'))throw Error('Previous-frame control failed');
 }
 const section=page.locator('section').first();const first=await section.locator('.counter').textContent();await section.locator('.play').click();
 await page.waitForFunction(text=>document.querySelector('.counter').textContent!==text,first);
 await section.locator('.play').click();
 if(errors.length)throw Error(errors.join('\n'));
 await fs.writeFile(path.join(out,'ui_validation.json'),JSON.stringify({images_loaded:234,sections:3,next_previous_pass:true,play_pause_pass:true,page_errors:errors},null,2));
 console.log('Review page: 234 images, 3 comparisons, frame controls and playback passed');
}finally{await browser.close();}

import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const {chromium}=await import(pathToFileURL('C:/Users/newtu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs'));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
try{
 const page=await browser.newPage({viewport:{width:1300,height:1050}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.join(here,'continuous_review_20261005_player.html')).href);
 const rows=[];
 for(const [word,count] of [['parents',31],['chicken',90],['class',121]]){
  await page.locator('#word').selectOption(word);
  await page.waitForFunction(n=>document.querySelector('#status').textContent===`โหลด ${n}/${n} ภาพ · ผิดพลาด 0`,count*3);
  await page.locator('#play').click();await page.waitForTimeout(250);
  const early=Number(await page.locator('#frame').inputValue());if(early<=0)throw Error('Playback did not advance');
  await page.screenshot({path:path.join(here,'continuous_review_20261005',word+'_early.png')});
  await page.waitForTimeout(300);
  const later=Number(await page.locator('#frame').inputValue());if(later<=early)throw Error('Playback stalled');
  await page.screenshot({path:path.join(here,'continuous_review_20261005',word+'_later.png')});
  await page.waitForFunction(()=>document.querySelector('#play').textContent==='เล่น');
  if(Number(await page.locator('#frame').inputValue())!==count-1)throw Error('Did not stop at final frame');
  await page.locator('#prev').click();if(Number(await page.locator('#frame').inputValue())!==count-2)throw Error('Step back failed');
  rows.push({word,early,later,frames:count,loaded:count*3,playback:true,step:true});
 }
 if(errors.length)throw Error(errors.join('\n'));
 await fs.writeFile(path.join(here,'continuous_review_20261005','ui_validation.json'),JSON.stringify(rows,null,2));console.log(rows);
}finally{await browser.close();}

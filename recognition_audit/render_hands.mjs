import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import http from 'node:http';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const auditOut=process.env.AUDIT_OUTPUT||path.join(root,'recognition_audit');
await fs.mkdir(auditOut,{recursive:true});
const {build}=await import(pathToFileURL(process.env.AUDIT_VITE||'C:/Users/newtu/signing-avatar-pose-animator/frontend-tooling/node_modules/vite/dist/node/index.js'));
const {chromium}=await import(pathToFileURL(process.env.AUDIT_PLAYWRIGHT||'C:/Users/newtu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs'));
const temp=await fs.mkdtemp(path.join(os.tmpdir(),'avatar-fidelity-'));
const hook=`window.runFidelityCase = async (payload, frameIndices) => {
 stopPlayback(); loadSequencePayload(payload);
 if(handRigReady || !showHandOverlays) throw new Error('Unexpected renderer mode');
 const results=[];
 for(const index of frameIndices){
  renderFrame(index);
  const frame=sequence.frames[index], anchor=getHipAnchor(frame.pose), hands={left:{},right:{}};
  for(const group of canvasScope.project.activeLayer.children){
   const side=group.data.auditSide;
   if(!side || !group.visible || group.opacity===0) continue;
   for(const stroke of group.children){
    const ids=stroke.data.auditIds; if(!ids) continue;
    stroke.segments.forEach((segment,i)=>{
     const p=stroke.localToGlobal(segment.point);
     const point=[(p.x-CANVAS_WIDTH*WAIST_CAMERA_TARGET_X)/WAIST_CAMERA_ZOOM+anchor.x,(p.y-CANVAS_HEIGHT*WAIST_CAMERA_TARGET_HIP_Y)/WAIST_CAMERA_ZOOM+anchor.y];
     const hand=frame.hands[side], raw=hand.keypoints[ids[i]].position, wrist=hand.keypoints[0].position;
     const bodyWrist=getPoseKeypoint(frame.pose,side==='left'?'leftWrist':'rightWrist');
     const target=bodyWrist && Number(bodyWrist.score)>=0.2 ? bodyWrist.position : wrist;
     const scale=Number(handOverlayScales[side]||1);
     const expected=[target.x+(raw.x-wrist.x)*scale,target.y+(raw.y-wrist.y)*scale];
     if(Math.hypot(point[0]-expected[0],point[1]-expected[1])>1e-6) throw new Error('Rendered endpoint / camera transform mismatch');
     hands[side][ids[i]]=point;
    });
   }
  }
  await new Promise(resolve=>requestAnimationFrame(resolve));
  results.push({hands,zoom:WAIST_CAMERA_ZOOM});
 }
 renderFrame(frameIndices[Math.floor(frameIndices.length/2)]);
 return results;
};`;
await build({configFile:false,root:path.join(root,'pose-animator'),base:'./',publicDir:false,resolve:{alias:{paper:path.join(root,'pose-animator/paper-global.js')}},build:{outDir:temp,emptyOutDir:false,rollupOptions:{input:path.join(root,'pose-animator/dataset_player.html')}},plugins:[{name:'audit-only',enforce:'pre',transform(source,id){
 if(!id.replaceAll('\\','/').split('?')[0].endsWith('/dataset_player.js'))return;
 source=source.replaceAll('\r\n','\n');
 function replace(a,b){if(source.split(a).length!==2)throw new Error('Hook not unique: '+a);source=source.replace(a,b);}
 replace('group.opacity = 1;', 'group.opacity = 1; group.data.auditSide = side;');
 replace('drawHandStroke(group, connection.a, connection.b, style.fill, connection.width);','drawHandStroke(group, connection.a, connection.b, style.fill, connection.width); group.lastChild.data.auditIds = [connection.a, connection.b].map(p => projected.findIndex(q => q === p));');
 replace('void pollMicReadiness();','/* audit: no microphone model startup */');
 replace('await startWordPreview(onGenerateFromText, loadBuiltInAvatar, BUILTIN_AVATARS);',hook);
 return {code:source,map:null};
}}]});
for(const sub of ['resources/illustration','vendor'])await fs.cp(path.join(root,'pose-animator',sub),path.join(temp,sub),{recursive:true});
const server=http.createServer(async(req,res)=>{try{
 const relative=decodeURIComponent(new URL(req.url,'http://localhost').pathname).replace(/^\/+/, '');
 const target=path.resolve(temp,relative);if(!target.startsWith(temp+path.sep))throw new Error('Invalid path');
 const data=await fs.readFile(target);res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml'})[path.extname(target)]||'application/octet-stream');res.end(data);
}catch{res.statusCode=404;res.end();}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
try{
 browser=await chromium.launch({headless:true,executablePath:process.env.AUDIT_BROWSER||'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(`http://127.0.0.1:${server.address().port}/dataset_player.html`);
 await page.waitForFunction(()=>typeof window.runFidelityCase==='function',null,{timeout:60000});
 const cases=JSON.parse(await fs.readFile(process.env.AUDIT_INPUTS||path.join(root,'recognition_audit/inputs.json'),'utf8')),results=[];
 for(const test of cases){
  const samples=await page.evaluate(({payload,frameIndices})=>window.runFidelityCase(payload,frameIndices),{payload:test.payload,frameIndices:test.frameIndices});
  results.push({phrase:test.phrase,samples});console.log(test.phrase,samples.length);
  await page.screenshot({path:path.join(auditOut,test.phrase.replaceAll(' ','_')+'.png')});
 }
 if(errors.length)throw new Error(errors.join('\n'));
 await fs.writeFile(path.join(auditOut,'rendered.json'),JSON.stringify(results));
}finally{if(browser)await browser.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
console.log('Audit build (production untouched):',temp);

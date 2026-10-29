import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import http from 'node:http';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const {build}=await import(pathToFileURL(process.env.AUDIT_VITE||'C:/Users/newtu/signing-avatar-pose-animator/frontend-tooling/node_modules/vite/dist/node/index.js'));
const {chromium}=await import(pathToFileURL(process.env.AUDIT_PLAYWRIGHT||'C:/Users/newtu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs'));
const out=process.env.AUDIT_OUTPUT||path.join(root,'recognition_audit/step1b'); await fs.mkdir(out,{recursive:true});
const temp=await fs.mkdtemp(path.join(os.tmpdir(),'avatar-fidelity-'));
const hook=`window.runFidelityCase = async (payload, frameIndices, options) => {
 window.auditOptions=options;
 window.auditPairTarget=(frame,rawWrist,fallback)=>{
  if(!options.pair) return fallback;
  const shifts=[];
  for(const side of ['left','right']){
   const hand=frame.hands?.[side],w=getHandPoint(hand,0),p=getPoseKeypoint(frame.pose,side==='left'?'leftWrist':'rightWrist');
   if(!w || getDetailedHandAlpha(hand)<=0 || !p || Number(p.score)<.2)return fallback;
   shifts.push({x:p.position.x-w.x,y:p.position.y-w.y});
  }
  return {x:rawWrist.x+(shifts[0].x+shifts[1].x)/2,y:rawWrist.y+(shifts[0].y+shifts[1].y)/2};
 };
 window.showAuditFrame = index => renderFrame(index);
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
     const body=bodyWrist && Number(bodyWrist.score)>=0.2 ? bodyWrist.position : wrist;
     const target=window.auditPairTarget(frame,wrist,{x:wrist.x+options.anchor*(body.x-wrist.x),y:wrist.y+options.anchor*(body.y-wrist.y)});
     const scale=1+options.scale*(Number(handOverlayScales[side]||1)-1);
     const expected=[target.x+(raw.x-wrist.x)*scale,target.y+(raw.y-wrist.y)*scale];
     if(Math.hypot(point[0]-expected[0],point[1]-expected[1])>1e-6) throw new Error('Rendered endpoint / camera transform mismatch');
     hands[side][ids[i]]=point;
    });
   }
  }
  if(index%20===0) await new Promise(resolve=>requestAnimationFrame(resolve));
  results.push({hands,zoom:WAIST_CAMERA_ZOOM});
 }
 renderFrame(frameIndices[Math.floor(frameIndices.length/2)]);
 return results;
};`;
await build({configFile:false,root:path.join(root,'pose-animator'),base:'./',publicDir:false,resolve:{alias:{paper:path.join(root,'pose-animator/paper-global.js')}},build:{outDir:temp,emptyOutDir:false,rollupOptions:{input:path.join(root,'pose-animator/dataset_player.html')}},plugins:[{name:'audit-only',enforce:'pre',transform(source,id){
 if(!id.replaceAll('\\','/').split('?')[0].endsWith('/dataset_player.js'))return;
 source=source.replaceAll('\r\n','\n');
 function replace(a,b){if(source.split(a).length!==2)throw new Error('Hook not unique: '+a);source=source.replace(a,b);}
 replace('const handScale = Number(handOverlayScales[side] || 1);', 'const handScale = 1 + window.auditOptions.scale * (Number(handOverlayScales[side] || 1) - 1);');
 replace('return projectHandPoint(rawPoint, rawWrist, poseWrist, handScale);', 'return projectHandPoint(rawPoint, rawWrist, window.auditPairTarget(frame,rawWrist,{x:rawWrist.x+window.auditOptions.anchor*(poseWrist.x-rawWrist.x), y:rawWrist.y+window.auditOptions.anchor*(poseWrist.y-rawWrist.y)}), handScale);');
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
 const variants=process.env.AUDIT_VARIANTS ? JSON.parse(process.env.AUDIT_VARIANTS) : [{id:'S0A50',scale:0,anchor:.5},{id:'S50A50',scale:.5,anchor:.5},{id:'S1A50',scale:1,anchor:.5},{id:'S0A1',scale:0,anchor:1},{id:'S50A1',scale:.5,anchor:1},{id:'S50A75',scale:.5,anchor:.75},{id:'S1A1',scale:1,anchor:1}];
 for(const options of variants) for(const test of cases){
  const samples=await page.evaluate(({payload,frameIndices,options})=>window.runFidelityCase(payload,frameIndices,options),{payload:test.payload,frameIndices:test.frameIndices,options});
  results.push({variant:options.id,phrase:test.phrase,samples});console.log(options.id,test.phrase,samples.length);
  if(process.env.AUDIT_CAPTURE_SEQUENCE==='1'){
   const folder=path.join(out,'frames',test.phrase);await fs.mkdir(folder,{recursive:true});
   for(let i=0;i<test.frameIndices.length;i++){
    await page.evaluate(index=>window.showAuditFrame(index),test.frameIndices[i]);
    await page.locator('canvas.illustration-canvas').screenshot({path:path.join(folder,String(i).padStart(4,'0')+'.png')});
   }
  }
  if(['hello','love','abdomen','hard','loan'].includes(test.phrase)) await page.screenshot({path:path.join(out,options.id+'_'+test.phrase.replaceAll(' ','_')+'.png')});
  if(process.env.AUDIT_REVIEW_FRAMES==='1') for(const fraction of [.25,.5,.75]){
   await page.evaluate(index=>window.showAuditFrame(index),test.frameIndices[Math.floor((test.frameIndices.length-1)*fraction)]);
   await page.screenshot({path:path.join(out,options.id+'_'+test.phrase.replaceAll(' ','_')+'_'+Math.round(fraction*100)+'.png')});
  }
 }
 if(errors.length)throw new Error(errors.join('\n'));
 await fs.writeFile(path.join(out,'rendered.json'),JSON.stringify(results));
}finally{if(browser)await browser.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
console.log('Audit build (production untouched):',temp);

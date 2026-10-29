// Read-only native/fallback audit + previously tested S50A1, isolated build.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let source=await fs.readFile(path.join(here,'step1b_render.mjs'),'utf8');
function replace(a,b){if(source.split(a).length!==2)throw Error('Nonunique hook '+a);source=source.replace(a,b);}
replace('const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));',`const root=${JSON.stringify(path.dirname(here))};`);
replace("if(!id.replaceAll('\\\\','/').split('?')[0].endsWith('/dataset_player.js'))return;",`
 if(id.replaceAll('\\\\','/').split('?')[0].endsWith('/illustrationGen/illustration.js')){
  const needle='scope.project.activeLayer.addChild(path);';
  if(source.split(needle).length!==2)throw Error('Native hook not unique');
  return {code:source.replace(needle,'if(skinnedPath.isHandPath && !skinnedPath.isProceduralHandPath) path.data.auditNativeSide=skinnedPath.handSide; '+needle),map:null};
 }
 if(!id.replaceAll('\\\\','/').split('?')[0].endsWith('/dataset_player.js'))return;`);
replace('results.push({hands,zoom:WAIST_CAMERA_ZOOM});',`
 const native={};
 const unproject=(x,y)=>[(x-CANVAS_WIDTH*WAIST_CAMERA_TARGET_X)/WAIST_CAMERA_ZOOM+anchor.x,(y-CANVAS_HEIGHT*WAIST_CAMERA_TARGET_HIP_Y)/WAIST_CAMERA_ZOOM+anchor.y];
 for(const side of ['left','right']){
  const paths=illustration.scope.project.activeLayer.children.filter(p=>p.data.auditNativeSide===side && p.visible && p.opacity>0);
  native[side]=paths.map(p=>{const b=p.bounds;const a=unproject(b.left,b.top),z=unproject(b.right,b.bottom);return {bounds:[...a,...z],opacity:p.opacity};});
 }
 const body=JSON.stringify(canvasScope.project.activeLayer.children.filter(p=>!p.data.auditNativeSide && !p.data.auditSide).map(p=>({opacity:p.opacity,segments:p.segments?.map(s=>[s.point.x,s.point.y,s.handleIn.x,s.handleIn.y,s.handleOut.x,s.handleOut.y])})));
 window.auditBodyRefs ||= new Map();const key=JSON.stringify(frame);
 if(window.auditBodyRefs.has(key) && window.auditBodyRefs.get(key)!==body)throw Error('Scale experiment changed non-hand geometry');
 window.auditBodyRefs.set(key,body);
 results.push({hands,native,index,zoom:WAIST_CAMERA_ZOOM});`);
// Save a few diagnostic actual Canvas frames, rather than synthetic diagrams.
replace("if(['hello','love','abdomen','hard','loan'].includes(test.phrase)) await page.screenshot({path:path.join(out,options.id+'_'+test.phrase.replaceAll(' ','_')+'.png')});",`
  if(['never','parents','hurry'].includes(test.label)){
   const folder=path.join(out,'frames',test.phrase+'__'+options.id);await fs.mkdir(folder,{recursive:true});
   const selected=[0,Math.floor(test.frameIndices.length/4),Math.floor(test.frameIndices.length/2),Math.floor(test.frameIndices.length*3/4),test.frameIndices.length-1];
   for(const i of selected){await page.evaluate(index=>window.showAuditFrame(index),test.frameIndices[i]);await page.locator('canvas.illustration-canvas').screenshot({path:path.join(folder,String(i).padStart(4,'0')+'.png')});}
  }`);
process.env.AUDIT_OUTPUT ||= path.join(here,'combined_winners200','renderer');
process.env.AUDIT_INPUTS ||= path.join(here,'combined_winners200','review_inputs.json');
process.env.AUDIT_VARIANTS ||= JSON.stringify([{id:'S1A1',scale:1,anchor:1},{id:'S50A1',scale:.5,anchor:1}]);
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

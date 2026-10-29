import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
let source=await fs.readFile(path.join(here,'fallback_boundary_render.mjs'),'utf8');
source=source.replace('const here=path.dirname(fileURLToPath(import.meta.url));',`const here=${JSON.stringify(here)};`);
const oldLine="const adjustment=await fs.readFile(path.join(here,'fallback_boundary_adjust.js'),'utf8');";
const bounds=`
let target=landmarkTarget;
if(window.auditOptions.renderBounds){
 const layer=canvasScope.project.activeLayer, count=layer.children.length;
 let temporary=[];
 try {
  drawHandDetails(sequence.frames[referenceIndex],side);
  temporary=layer.children.slice(count);
  for(const q of temporary)q.scale(WAIST_CAMERA_ZOOM,new canvasScope.Point(0,0));
  const bounds=temporary.filter(q=>q.data.auditSide===side && q.visible && q.opacity>0).map(q=>q.strokeBounds);
  if(!bounds.length)throw Error('Reference detailed drawing missing');
  target=Math.hypot(Math.max(...bounds.map(b=>b.right))-Math.min(...bounds.map(b=>b.left)),Math.max(...bounds.map(b=>b.bottom))-Math.min(...bounds.map(b=>b.top)))/WAIST_CAMERA_ZOOM;
 } finally {for(const q of layer.children.slice(count))q.remove();}
 if(layer.children.length!==count)throw Error('Reference measurement changed layer');
 const clones=paths.map(q=>q.clone({insert:false}));
 try {
  for(const q of clones)q.scale(WAIST_CAMERA_ZOOM,new canvasScope.Point(0,0));
  const bs=clones.map(q=>q.strokeBounds);
  diagonal=Math.hypot(Math.max(...bs.map(b=>b.right))-Math.min(...bs.map(b=>b.left)),Math.max(...bs.map(b=>b.bottom))-Math.min(...bs.map(b=>b.top)))/WAIST_CAMERA_ZOOM;
 } finally {for(const q of clones)q.remove();}
}
`;
source=source.replace(oldLine,`let adjustment=await fs.readFile(path.join(here,'fallback_boundary_adjust.js'),'utf8');
adjustment=adjustment.replace('let reference=null, distance=0;','let reference=null, distance=0, referenceIndex=-1;');
adjustment=adjustment.replace('reference=h;distance=d;break;','reference=h;referenceIndex=j;distance=d;break;');
adjustment=adjustment.replace('const target=Math.hypot','const landmarkTarget=Math.hypot');
adjustment=adjustment.replace('const diagonal=Math.hypot','let diagonal=Math.hypot');
adjustment=adjustment.replace('if(diagonal<1e-6 || target<1e-6)continue;',${JSON.stringify(bounds)}+'\\nif(diagonal<1e-6 || target<1e-6)continue;');
adjustment=adjustment.replace('const rotation=Math.max(-45,Math.min(45,angle))*weight;','const rotation=0;');
adjustment=adjustment.replace('{distance,rotation,scale}','{distance,rotation,scale,referenceIndex,target,landmarkTarget}');`);
source=source.replaceAll('fallback_boundary_20261003','render_bounds_scale_camera_20261004');
source=source.replace("[{id:'OFF',scale:1,anchor:1,fallbackAdjust:false},{id:'ON',scale:1,anchor:1,fallbackAdjust:true}]","[{id:'OFF',scale:1,anchor:1,fallbackAdjust:false},{id:'OLD',scale:1,anchor:1,fallbackAdjust:true},{id:'NEW',scale:1,anchor:1,fallbackAdjust:true,renderBounds:true}]");
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

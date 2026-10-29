// Four-cell ablation using production imports in a temporary bundle only.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {installFallbackAudit,capFactor} from './fallback_cap.mjs';
import assert from 'node:assert/strict';
assert.equal(capFactor(100,50),.5);assert.equal(capFactor(25,50),1);
assert.equal(capFactor(100,null),1);assert.equal(capFactor(0,50),1);
const here=path.dirname(fileURLToPath(import.meta.url));
let source=await fs.readFile(path.join(here,'step1b_render.mjs'),'utf8');
function replace(a,b){if(source.split(a).length!==2)throw Error('Nonunique hook: '+a);source=source.replace(a,b);}
replace('const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));',`const root=${JSON.stringify(path.dirname(here))};`);
replace('window.auditOptions=options;',`window.auditOptions=options;window.auditPayload=payload;(${installFallbackAudit.toString()})();`);
replace("if(!id.replaceAll('\\\\','/').split('?')[0].endsWith('/dataset_player.js'))return;",`
 if(id.replaceAll('\\\\','/').split('?')[0].endsWith('/illustrationGen/illustration.js')) {
  const needle='scope.project.activeLayer.addChild(path);';
  if(source.split(needle).length!==2) throw Error('Native path hook mismatch');
  return {code:source.replace(needle, "if(skinnedPath.isHandPath && !skinnedPath.isProceduralHandPath) path.data.auditNativeSide=skinnedPath.handSide; " + needle),map:null};
 }
 if(!id.replaceAll('\\\\','/').split('?')[0].endsWith('/dataset_player.js'))return;`);
replace("replace('void pollMicReadiness();'",`replace('  illustration.draw();', '  illustration.draw(); window.auditFallback(illustration, frame, handOverlayScales, window.auditOptions, frameIndex);');
 replace('void pollMicReadiness();'`);
replace('results.push({hands,zoom:WAIST_CAMERA_ZOOM});',`canvasScope.view.update();
 const original=document.querySelector('canvas.illustration-canvas');
 const exportCanvas=document.createElement('canvas');exportCanvas.width=original.width;exportCanvas.height=original.height;
 const ctx=exportCanvas.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,exportCanvas.width,exportCanvas.height);ctx.drawImage(original,0,0);
 results.push({hands,zoom:WAIST_CAMERA_ZOOM,fallback:window.auditFallbackRecord,png:exportCanvas.toDataURL('image/png').split(',')[1]});`);
replace('results.push({variant:options.id,phrase:test.phrase,samples});',`const folder=path.join(out,'frames',options.id,test.phrase);await fs.mkdir(folder,{recursive:true});
 for(let i=0;i<samples.length;i++){await fs.writeFile(path.join(folder,String(i).padStart(4,'0')+'.png'),Buffer.from(samples[i].png,'base64'));delete samples[i].png;}
 results.push({variant:options.id,phrase:test.phrase,samples});`);
process.env.AUDIT_OUTPUT=process.env.FALLBACK_OUTPUT||path.join(here,'fallback_trial');
process.env.AUDIT_INPUTS=path.join(here,'method_review/inputs.json');
process.env.AUDIT_VARIANTS=process.env.FALLBACK_VARIANTS||JSON.stringify([
 {id:'baseline',scale:1,anchor:1,cap:false},
 {id:'scale_only',scale:.5,anchor:1,cap:false},
 {id:'cap_only',scale:1,anchor:1,cap:true},
 {id:'combined',scale:.5,anchor:1,cap:true}]);
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

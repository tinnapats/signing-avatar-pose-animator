import fs from 'node:fs/promises';import path from 'node:path';import crypto from 'node:crypto';
const root='C:/Users/newtu/signing-avatar-github-newton-upload',here=path.join(root,'recognition_audit'),out=path.join(here,'pixels30_20261007');
await fs.mkdir(out,{recursive:true});
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const source=await read(path.join(here,'slgcn200_20261002/protocol.json'));const labels=source.labels.slice(0,30);
const descriptors=[];for(let i=0;i<30;i++){const file=path.join(here,'slgcn200_20261002/cases',String(i).padStart(3,'0')+'.json'),raw=await fs.readFile(file);const c=JSON.parse(raw);if(c.label!==labels[i])throw Error('Labels mismatch');descriptors.push({id:String(i).padStart(3,'0'),label:c.label,file,sha256:sha(raw),frames:c.frameIndices.length});}
const production={};for(const f of ['pose-animator/dataset_player.js','pose-animator/illustrationGen/illustration.js','pose-animator/illustrationGen/skeleton.js','export_pose_animator_sequence.py'])production[f]=sha(await fs.readFile(path.join(root,f)));
const protocol={labels,descriptors,production,fps:30,selection:'First30 in frozen200 order, no selection by new detector/model scores',pipeline:source.pipeline,
 detector:{primary:'MediaPipe Holistic tracking',secondary:'Holistic + full-frame Hands only for missing hand; non-selfie handedness swapped; confidence >=.5',model_complexity:1,min_detection_confidence:.5,min_tracking_confidence:.5},
 missing:'No filling or hold. Missing coordinates zero before upstream normalization. Report coverage with scores; no rejection based on low score.',
 input:'Only PNG RGB white-composited frames to detector. No payload, renderer endpoints, labels or crops to detector. Labels only for evaluation.',
 model:'Existing unchanged ST-GCN, 27 x/y points, shoulder clip normalization, 2000 unrestricted classes',
 controls:'26 blank frames + first26 frames of known human clip for detector; whole-frame shuffle and zero hand blocks for primary ST-GCN',
 scope:'Pilot on known development30, not unseen human ASL evaluation. Use cached no_hold_no_smoothing payload and current renderer; not regeneration from live UI.'};
const protocolPath=path.join(out,'protocol.json');try{if(JSON.stringify(await read(protocolPath))!==JSON.stringify(protocol))throw Error('Frozen protocol differs');}catch(e){if(e.code==='ENOENT')await fs.writeFile(protocolPath,JSON.stringify(protocol,null,2));else throw e;}
let code=await fs.readFile(path.join(here,'render_hands.mjs'),'utf8');code=code.replace('const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));',`const root=${JSON.stringify(root)};`);
const start=code.indexOf('const hook=`'),end=code.indexOf('await build(',start);if(start<0||end<0)throw Error('Hook missing');
const hook=`window.runFidelityCase=async(payload,indices)=>{stopPlayback();loadSequencePayload(payload);if(handRigReady||!showHandOverlays)throw Error('Unexpected hand mode');const images=[];for(const index of indices){renderFrame(index);canvasScope.view.update();const canvas=document.querySelector('canvas.illustration-canvas');images.push(canvas.toDataURL('image/png').split(',')[1]);}const c=document.querySelector('canvas.illustration-canvas');return {images,width:c.width,height:c.height};};`;
code=code.slice(0,start)+'const hook='+JSON.stringify(hook)+';\n'+code.slice(end);
const ls=code.indexOf(' const cases=JSON.parse('),le=code.indexOf(' if(errors.length)',ls);if(ls<0||le<0)throw Error('Loop missing');
code=code.slice(0,ls)+` const descriptors=${JSON.stringify(descriptors)},results=[];
 for(const d of descriptors){const folder=path.join(auditOut,'frames',d.id);await fs.mkdir(folder,{recursive:true});const saved=path.join(auditOut,'capture_'+d.id+'.json');try{const r=JSON.parse(await fs.readFile(saved));results.push(r);continue;}catch(e){if(e.code!=='ENOENT')throw e;}
 const test=JSON.parse(await fs.readFile(d.file,'utf8'));const capture=await page.evaluate(({payload,indices})=>window.runFidelityCase(payload,indices),{payload:test.payload,indices:test.frameIndices});
 const entries=[];for(let i=0;i<capture.images.length;i++){const f=String(i).padStart(4,'0')+'.png',bytes=Buffer.from(capture.images[i],'base64');await fs.writeFile(path.join(folder,f),bytes);entries.push({file:f,sourceIndex:test.frameIndices[i],time:i/30,sha256:(await import('node:crypto')).createHash('sha256').update(bytes).digest('hex')});}
 const r={id:d.id,label:d.label,frames:entries,width:capture.width,height:capture.height,fps:30};await fs.writeFile(saved,JSON.stringify(r));results.push(r);console.log('captured',d.id,d.label,entries.length,capture.width,capture.height);
 }
`+code.slice(le);code=code.replace("path.join(auditOut,'rendered.json')","path.join(auditOut,'capture_manifest.json')");
process.env.AUDIT_OUTPUT=out;await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
for(const[f,h]of Object.entries(production))if(sha(await fs.readFile(path.join(root,f)))!==h)throw Error('Production changed '+f);

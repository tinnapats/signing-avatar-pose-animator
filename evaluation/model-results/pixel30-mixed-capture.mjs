import fs from 'node:fs/promises';
let code=await fs.readFile(new URL('./pixel30-coordinate-capture.mjs',import.meta.url),'utf8');
code=code.replace('rows.push({index,hands,pixelRatio:',`const anchor=getHipAnchor(sequence.frames[index].pose)||{x:CANVAS_WIDTH*.5,y:CANVAS_HEIGHT*.72};const map=(x,y)=>{const p=canvasScope.view.projectToView(new canvasScope.Point((x-anchor.x)*WAIST_CAMERA_ZOOM+CANVAS_WIDTH*WAIST_CAMERA_TARGET_X,(y-anchor.y)*WAIST_CAMERA_ZOOM+CANVAS_HEIGHT*WAIST_CAMERA_TARGET_HIP_Y));return [p.x,p.y];};rows.push({index,hands,affine:[map(0,0),map(1,0),map(0,1)],pixelRatio:`);
code=code.replaceAll('coordinate_reference.json','mixed_reference.json');
await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

import fs from 'node:fs/promises';import path from 'node:path';
const out='C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007';const read=async f=>JSON.parse(await fs.readFile(path.join(out,f),'utf8'));
const detection=await read('detection_summary.json'),result=await read('results.json'),protocol=await read('protocol.json'),caps=await read('capture_manifest.json'),controls=await read('controls.json');
const sum=k=>detection.per_clip.reduce((s,r)=>s+r[k],0),n=sum('frames'),pct=(v,total=n)=>(100*v/total).toFixed(2)+'%';
const names={pixels_holistic:'ภาพ → Holistic → ST-GCN (หลัก)',pixels_holistic_hands:'ภาพ → Holistic + Hands เมื่อมือหาย → ST-GCN',pixels_shuffled:'ภาพ → Holistic → สลับลำดับเฟรม',pixels_zero_hands:'ภาพ → Holistic → ลบพิกัดมือก่อนโมเดล',archived_pre:'พิกัดก่อน Renderer เดิม (อ้างอิง)',archived_post:'พิกัดมือ Renderer + ลำตัว payload เดิม (อ้างอิง)'};
const order=['pixels_holistic','pixels_holistic_hands','pixels_shuffled','pixels_zero_hands','archived_pre','archived_post'];
const lines=['# ทดลองภาพ Animation → พิกัด → ST-GCN: 30 คำ','',
'7 ตุลาคม 2026 — ใช้ 30 คำแรกตามลำดับชุด 200 คำเดิม ตรึงก่อนดูผลรอบนี้ หนึ่งคลิปต่อคำ เป็นชุดพัฒนาเดิม ไม่ใช่ held-out ใหม่',
'','## วิธีและขอบเขต',
`จับภาพ Canvas ใหม่ครบ ${n} เฟรมที่ 30 FPS ขนาด 808×808 ตามเวลาเดิม จาก payload no_hold_no_smoothing ที่บันทึกไว้ ใช้ Renderer ปัจจุบัน ไม่ใช่การสร้าง payload ใหม่จากระบบสด`,
'ป้อน RGB ของ Canvas ที่วางบนพื้นขาวให้ detector โดยไม่ให้พิกัด payload, endpoints, คำตอบ หรือกรอบครอปจาก Renderer ตรวจด้วย MediaPipe Holistic complexity1 และ Hands complexity1, confidence .5, tracking, reset ทุกคลิป',
'เก็บ pose33, pose-world, face468 และมือ21จุดต่อข้างเมื่อ detector คืนข้อมูล พร้อม x/y/z/visibility, Hands handedness และเวลาแต่ละเฟรม ไม่มีการรับรองว่าทุกจุดตรวจถูก',
'ผลหลักใช้ Holistic ส่วนผลเสริมเติมเฉพาะมือที่ Holistic ไม่พบด้วย Hands แบบภาพเต็ม กำหนดกลับ handedness สำหรับภาพที่ไม่ใช่ selfie และเลือก confidence สูงสุดต่อข้าง ไม่เติมข้อมูลข้ามเฟรม',
'แปลงพิกัด normalized ภาพเป็น pixel x/y เลือก27จุด และใช้ shoulder normalization เดิม โมเดลไม่รับ z/face468 ทั้งหมด มือที่หายใช้ศูนย์ก่อน normalization ตามวิธีเดิม ไม่มีการนำจุดจาก payload มาช่วย',
'','## ผลคะแนนโมเดล',
'| เงื่อนไข | ถูก/30 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top-5 Accuracy |',
'|---|---:|---:|---:|---:|---:|---:|'];
for(const key of order){const r=result.summary[key];lines.push(`| ${names[key]} | ${r.correct}/30 | ${r.accuracy.toFixed(2)}% | ${r.precision.toFixed(2)}% | ${r.recall.toFixed(2)}% | ${r.f1.toFixed(2)}% | ${r.top5.toFixed(2)}% |`);}
lines.push('','Macro เฉลี่ย 30 true labels, zero_division=0; โมเดลเลือกได้2000คำ คำตอบนอกชุดยังผิด ผลอ้างอิงใช้รายคำเดิมของ30คำเดียวกันจาก slgcn200 ไม่ใช่การรันใหม่ ไม่เทียบกับคะแนนเก่าชุด30ที่รายชื่ออาจต่างกัน',
'','## การตรวจพบจากภาพ (coverage ไม่ใช่ accuracy ของข้อต่อ)',
'| รายการ | เฟรม | สัดส่วน |','|---|---:|---:|');
for(const [k,l] of [['pose','Holistic พบลำตัว'],['left','Holistic พบมือซ้าย'],['right','Holistic พบมือขวา'],['any_hand','Holistic พบอย่างน้อยหนึ่งมือ'],['both_hands','Holistic พบสองมือ'],['secondary_any_hand','หลังเสริม Hands พบอย่างน้อยหนึ่งมือ'],['secondary_both_hands','หลังเสริม Hands พบสองมือ']])lines.push(`| ${l} | ${sum(k)}/${n} | ${pct(sum(k))} |`);
lines.push('','ไม่ใช้การพบสองมือทุกเฟรมเป็นเงื่อนไขตัดคลิป เพราะท่าบางคำใช้มือเดียว/มีการบังกัน การพบจุดมากขึ้นยังไม่ยืนยันว่าตำแหน่งนิ้วถูกต้อง',
'','## ชุดควบคุม detector',`- วิดีโอมนุษย์26เฟรม: pose ${controls.human.pose}/26, อย่างน้อยหนึ่งมือ ${controls.human.any_hand}/26, สองมือ ${controls.human.both_hands}/26`,
`- ภาพขาว26เฟรม: pose ${controls.blank.pose}/26, อย่างน้อยหนึ่งมือ ${controls.blank.any_hand}/26, Hands พบ ${controls.blank.hands_detector_any}/26`,
'','## ผลรายคำ',
'| คำ | เฟรม | มือ≥1 Holistic | มือ2 Holistic | มือซ้ายหายยาวสุด (เฟรม) | มือขวาหายยาวสุด | ทายหลัก | ทายเสริม |',
'|---|---:|---:|---:|---:|---:|---|---|');
for(const r of detection.per_clip){const pred=v=>result.predictions.find(p=>p.label===r.label&&p.variant===v).prediction;lines.push(`| ${r.label} | ${r.frames} | ${pct(r.any_hand,r.frames)} | ${pct(r.both_hands,r.frames)} | ${r.longest_no_left} | ${r.longest_no_right} | ${pred('pixels_holistic')} | ${pred('pixels_holistic_hands')} |`);}
lines.push('','## ตรวจภาพด้วยตาและข้อสรุปเบื้องต้น',
'ตรวจ contact sheet ของ hello, please และ hard: hello/please มีช่วงที่ detector วางจุดมือใกล้มือที่วาด แต่ไม่ได้ยืนยันความถูกต้องทุกข้อต่อ ส่วน hard มีเฟรมที่แนวแขนที่ตรวจพบเบี่ยงไปบริเวณคอ/ลำตัวและไม่พบจุดมือ แม้ภาพยังมีมืออยู่ นี่เป็นหลักฐานว่าการอ่านภาพมีข้อผิดพลาด ไม่ใช่หลักฐานว่าท่าภาษามือผิดทั้งหมด',
'ผลหลักและผลเสริมทายถูกเฉพาะ please การเพิ่ม coverage ด้วย Hands ยังไม่เพิ่มจำนวนคำที่ทายถูก จึงยังไม่ควรขยายเป็น200คำเพื่อใช้ตัดสินคุณภาพท่า ควรตรวจ mapping ซ้าย/ขวาและความตรงของพิกัดจากภาพกับจุดที่วาด ในชุดย่อยที่ตรึงไว้ก่อน โดยไม่ปรับวิธีตามคำตอบรายคำ',
'หน้า review.html ผ่านการทดสอบเปิดภาพ808×808 มีตัวเลือกครบ30คำ สลับคำ เลื่อนเฟรม และเปิด overlay Hands ได้โดยไม่มี JavaScript error การตรวจภาพด้วยตายังไม่ได้ทำครบทุกเฟรม',
'','## ข้อจำกัดและการตีความ',
'คะแนนรอบนี้รวมความผิดพลาดการตรวจจับจากภาพกับการรู้จำคำของ ST-GCN จึงแยกสาเหตุไม่ได้จากคะแนนคำอย่างเดียว Detector ฝึกกับภาพมนุษย์แต่ใช้ภาพ Avatar รูปร่างและการบังมืออาจทำให้วางข้อต่อผิด',
'Hands เสริมอาจตรวจมือเดียวซ้ำสองช่องหรือสลับซ้ายขวาได้ ผลเสริมจึงเป็น diagnostic ที่ต้องตรวจภาพ ไม่ควรเลือกใช้เพียงเพราะ coverage สูงกว่า',
'เก็บนิ้ว21จุดไว้ตรวจภาพ แต่ ST-GCN ใช้มือข้างละ10จุด จึงยังวัดนิ้วที่ไม่ได้ป้อนเข้าโมเดลไม่ได้ ไม่มีผู้รู้ ASL ยืนยัน และยังไม่ยืนยันการไม่ซ้ำข้อมูลฝึก',
'','## ไฟล์ตรวจย้อนหลัง',
'- review.html: เลือกคำ เลื่อนดูทุกเฟรม เปิด/ปิด overlay ของ detector และดูคำทำนาย',
'- review/*.png: ทุกคำมี6เฟรมเวลาเท่าๆกัน ภาพบน Canvas ภาพล่าง Holistic overlay',
'- frames/: PNG ทุกเฟรม; capture_manifest.json เก็บ hash ขนาดภาพและเฟรมต้นทาง',
'- detected/: raw landmarks ทุกเฟรม รวม face/pose-world และ detectorสองตัว',
'- arrays/: พิกัดที่เตรียมให้โมเดล primary/secondary ก่อนเลือก27จุดและnormalize',
'- results.json: ผลโมเดลรายคำทั้ง Top-1/Top-5/target probability และคะแนนรวม',
'- protocol.json: กติกาที่ตรึงและhash source; production fingerprint ตรวจหลังรันตรงเดิม','');
await fs.writeFile(path.join(out,'REPORT_TH.md'),lines.join('\n'));
const clips=[];for(const cap of caps){const raw=await read('detected/'+cap.id+'.json');clips.push({id:cap.id,label:cap.label,frames:cap.frames.map(x=>x.file),rows:raw.rows.map(r=>({pose:r.pose,left:r.left,right:r.right,hands:r.hands,handedness:r.handedness})),predictions:result.predictions.filter(r=>r.label===cap.label)});}
await fs.writeFile(path.join(out,'review-data.js'),'window.pixelAudit='+JSON.stringify({clips,summary:result.summary})+';');
await fs.writeFile(path.join(out,'review.html'),`<!doctype html><html lang="th"><meta charset="utf-8"><title>Pixels to ST-GCN — 30 words</title><style>body{font:17px Tahoma,sans-serif;color:#163047;background:#f0f4f8;max-width:1400px;margin:24px auto;padding:0 20px}h1{font-size:27px}.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}select,button{font:inherit;padding:8px}#stage{position:relative;max-width:700px;aspect-ratio:1;background:white}#stage img,#stage canvas{position:absolute;inset:0;width:100%;height:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}input[type=range]{width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #ccd8df}p{line-height:1.6}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><h1>ภาพ Animation → จุดพิกัดจากภาพ → ST-GCN</h1><p>30 คำ · 30 FPS · สีเทา: pose · น้ำเงิน: มือซ้าย Holistic · ส้ม: มือขวา Holistic · ม่วง: Hands แยก · คะแนนเป็นผลนำร่อง</p><div class="controls"><select id="word"></select><button id="play">เล่น</button><label><input type="checkbox" id="hol" checked>Holistic</label><label><input type="checkbox" id="hands">Hands แยก</label></div><div class="grid"><div><div id="stage"><img id="img"><canvas id="overlay" width="808" height="808"></canvas></div><input id="frame" type="range" min="0" value="0"><p id="status"></p></div><div><h2>ผลทำนายคำ</h2><table id="pred"></table><p>ข้อมูล overlay มาจากภาพเท่านั้น เมื่อไม่พบมือ ระบบใช้ค่าศูนย์ก่อน normalization ไม่มีการเติมจุดจาก payload</p><p>หน้าเว็บแสดงพิกัดบนภาพทุกเฟรม; ไฟล์ detected เก็บ x/y/z/visibility และ face landmarks เพิ่มเติม</p><a href="REPORT_TH.md">รายงานฉบับเต็ม</a></div></div><script src="review-data.js"></script><script>
const $=id=>document.getElementById(id),data=window.pixelAudit,sel=$('word'),range=$('frame'),ctx=$('overlay').getContext('2d');let timer=null;
data.clips.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.label;sel.appendChild(o)});
const edges=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]],body=[[0,2],[0,5],[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]];
function draw(points,es,col){if(!points.length)return;ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=2;for(const[a,b]of es){ctx.beginPath();ctx.moveTo(points[a][0]*808,points[a][1]*808);ctx.lineTo(points[b][0]*808,points[b][1]*808);ctx.stroke()}for(const p of points){ctx.beginPath();ctx.arc(p[0]*808,p[1]*808,3,0,Math.PI*2);ctx.fill()}}
function show(){const c=data.clips[+sel.value],i=+range.value,r=c.rows[i];$('img').src='frames/'+c.id+'/'+c.frames[i];ctx.clearRect(0,0,808,808);if($('hol').checked){draw(r.pose,body,'#778899');draw(r.left,edges,'#397bbe');draw(r.right,edges,'#ee7c2d')}if($('hands').checked)for(const h of r.hands)draw(h,edges,'#9a39b9');$('status').textContent='เฟรม '+i+' / '+(c.rows.length-1)+' · เวลา '+(i/30).toFixed(2)+' วินาที · pose '+r.pose.length+' จุด · มือซ้าย '+r.left.length+' จุด · มือขวา '+r.right.length+' จุด · Hands '+r.hands.length+' มือ';}
function select(){if(timer){clearInterval(timer);timer=null;$('play').textContent='เล่น'}const c=data.clips[+sel.value];range.max=c.rows.length-1;range.value=0;$('pred').replaceChildren();for(const r of c.predictions){const tr=document.createElement('tr');for(const s of [r.variant,r.prediction+(r.prediction===c.label?' ✓':' ✗'),r.top5.join(', ')]){const td=document.createElement('td');td.textContent=s;tr.appendChild(td)}$('pred').appendChild(tr)}show()}
sel.onchange=select;range.oninput=show;$('hol').onchange=show;$('hands').onchange=show;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='เล่น'}else{$('play').textContent='หยุด';timer=setInterval(()=>{range.value=(+range.value+1)%(+range.max+1);show()},1000/30)}};select();
</script></html>`);
console.log(JSON.stringify({frames:n,pose:sum('pose'),any:sum('any_hand'),both:sum('both_hands'),secondaryBoth:sum('secondary_both_hands'),scores:result.summary,controls:{human:controls.human,blank:controls.blank}},null,2));

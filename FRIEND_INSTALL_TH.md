# คู่มือติดตั้งบนเครื่องเพื่อน — branch newton

## ชุดนี้ทำอะไรได้

1. เปิดเว็บ รับข้อความ สร้าง Animation จาก SQLite และรับเสียงด้วย XLSR-53 English
2. รัน ST-GCN/SL-GCN ใหม่จากข้อมูลทดลองเดิม 200 คลิป (Experiment 1–3)
3. มีโค้ดสร้างข้อมูลและตรวจความตรงเชิงพิกัดมือหลัง Renderer สามารถติดตั้งเครื่องมือเสริมแล้วรันได้

**ไม่ใช่** ชุดรวมทุกการทดลองในอดีต: ไม่มีภาพ/วิดีโอและโมเดล RTMW/SDPose/MediaPipe ทุกเวอร์ชัน
Experiment 2–3 ในตัวเปิดนี้อ่านพิกัดที่เก็บไว้ ไม่ได้ตรวจภาพใหม่ ส่วน Experiment 3 มี Renderer ช่วย ไม่ใช่ image-only

## A. รับและแตกไฟล์

- ส่ง ZIP ใน `friend_bundle/` พร้อมไฟล์ `.sha256.txt` ให้เพื่อนผ่านช่องทางส่วนตัว
- ZIP เป็นสำเนาเนื้อไฟล์จริง ไม่ใช่ hard link หรือ shortcut ไปยังเครื่องต้นทาง
- แตกลงที่ใดก็ได้ เช่น `D:\SignProject\signing-avatar-newton` ห้ามรันภายใน ZIP
- ต้องมีพื้นที่สำหรับ ZIP, ไฟล์ที่แตก และไลบรารี Python (ควรเผื่อหลาย GB เพิ่มเติม)
- โฟลเดอร์นี้มีโค้ดจาก Git แล้ว ไม่จำเป็นต้อง clone ซ้ำ ไม่มี `.git` หรือ environment ส่วนตัวรวมไปด้วย
- ถ้าเพื่อน clone branch newton ไว้แล้ว ให้คัดลอกเฉพาะ `SLclean.sqlite3`, `models/`, โมเดล ZIP ใน `recognition_audit/`, โฟลเดอร์ `slgcn200_20261002/cases/` และ `evaluation/model-results/oracle200-20261011/` จากชุดนี้ไปตำแหน่งเดิม ไม่คัดลอก `.venv` จากเครื่องอื่น
- สำหรับการแจกจ่ายต่อ ต้องตรวจสิทธิ์ของฐานข้อมูล WLASL/SLclean และโมเดลต้นทางแยกจาก license ของโค้ด ไม่ถือว่าไฟล์เหล่านี้ได้รับสิทธิ์เผยแพร่สาธารณะทั้งหมด

## B. ติดตั้ง

1. ติดตั้ง Python 3.12 แบบ 64-bit พร้อม Python launcher (`py`)
2. เปิด `setup_friend.cmd` โดยไม่ต้องใช้สิทธิ์ Administrator
3. ครั้งแรกต้องมีอินเทอร์เน็ตเพื่อดาวน์โหลดไลบรารี Python ตัวติดตั้งสร้าง `.venv` ในโฟลเดอร์นี้เอง
4. เมื่อจบจะแสดงผลตรวจไฟล์และไลบรารี หากล้มเหลวให้อ่านข้อผิดพลาด ไม่ข้ามขั้นตรวจ

`requirements-handoff.txt` ระบุเวอร์ชันที่บันทึกจากเครื่องต้นทาง ไม่ใช่การรับรองว่าเครื่อง Windows ทุกเครื่องติดตั้งได้สำเร็จ หาก PyPI/เครือข่ายหรือนโยบายเครื่องบล็อกเวอร์ชันนั้น ต้องแก้สภาพแวดล้อมก่อน ไม่ควรเปลี่ยนเวอร์ชันเงียบ ๆ แล้วอ้างผลเหมือนเดิม

ตรวจไฟล์หลังแตก ZIP ก่อนแก้โค้ด (ใช้เวลาสักครู่):

```powershell
py -3.12 verify_friend_bundle.py
```

ตรวจส่วนประกอบหลังติดตั้ง:

```powershell
.\.venv\Scripts\python.exe -B check_friend_setup.py
```

## C. ใช้ระบบหลัก

ดับเบิลคลิก `open_signing_avatar.cmd` แล้วใช้งานเว็บที่เปิดขึ้น
ปกติคือ `http://127.0.0.1:8025/dataset_player.html` หาก port ถูกใช้ เซิร์ฟเวอร์อาจเลือก port ถัดไป ให้ดู URL ที่แสดง
ทดสอบพิมพ์ `hello`, `love`, `abdomen` ก่อน แล้วจึงอนุญาตไมโครโฟนในเบราว์เซอร์
โมเดลเสียงอ่านจาก `models/xlsr53-english` ในชุดส่งต่อ ไม่ต้องพึ่ง cache ของบัญชีผู้ใช้เดิม และไม่ดาวน์โหลดน้ำหนักตอนกดไมค์
ครั้งแรกยังมีเวลาโหลดโมเดลเข้า RAM; รอสถานะพร้อมก่อนพูดภาษาอังกฤษ
ปิดหน้าต่างเซิร์ฟเวอร์เมื่อเลิกใช้ ตัวประเมินไม่ต้องเปิดเซิร์ฟเวอร์นี้

## D. ใช้ตัวประเมิน

ดับเบิลคลิก `run_evaluation.cmd` แล้วเลือก 1, 2 หรือ 3
ตัวเปิดหา `.venv` จากโฟลเดอร์โปรเจกต์ ไม่ต้องมีชื่อผู้ใช้หรือไดรฟ์เหมือนเครื่องต้นทาง
ผลใหม่บันทึกใน `evaluation/runs/<เวลา>/results.json` ไม่ทับผลเดิม

```powershell
.\run_evaluation.cmd --experiment 1 --model stgcn --limit 200
.\run_evaluation.cmd --experiment 1 --model slgcn --limit 200
.\run_evaluation.cmd --experiment 2 --limit 200
.\run_evaluation.cmd --experiment 3 --limit 200
```

ค่าตรวจสอบจากรอบก่อนย้ายเครื่อง: Exp1 ST-GCN 122/200, SL-GCN 96/200; Exp2 RTMW baseline 19/200; Exp3 renderer-assisted 99/200
นี่คือคะแนนรู้จำคำ ไม่ใช่เปอร์เซ็นต์ความถูกต้องทางภาษามือหรือความเข้าใจของผู้ใช้ และ Exp2 นี้ไม่ใช่วิธีที่ดีที่สุดของทุกการทดลอง
หากใช้ environment ของตัวเอง ให้ตั้ง `SIGN_PYTHON` หรือ `SIGN_EVAL_PYTHON` เป็นเส้นทาง python.exe โดย `SIGN_EVAL_PYTHON` มีลำดับเหนือ `SIGN_PYTHON` สำหรับตัวประเมิน

## E. เสริม: ตรวจพิกัดหลัง Renderer ใหม่

ส่วนนี้ไม่ต้องใช้โมเดล RTMW แต่ต้องมี Node.js ที่รองรับ Vite 7 และติดตั้ง browser สำหรับ Playwright:

```powershell
cd frontend-tooling
npm install
npm install --no-save playwright
npx playwright install chromium
cd ..
.\.venv\Scripts\python.exe -B measure_render_fidelity.py prepare
$env:FIDELITY_OUTPUT = Join-Path (Get-Location).Path ('fidelity_verification/run-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
node verify_render_fidelity.mjs
.\.venv\Scripts\python.exe -B summarize_fidelity_verification.py
```

`prepare` สร้างข้อมูลใหม่จาก SQLite และเขียน `fidelity_results/inputs.json` ให้สำรองข้อมูลเก่าก่อนรันซ้ำ
ค่าเริ่มต้น source คือโปรเจกต์ที่ติดตั้งอยู่; หากกำหนด `FIDELITY_SOURCE` ต้องใช้ค่าเดียวกันตลอดการจับภาพและสรุปผล
ตัวตรวจใช้ temporary build ไม่แก้ Renderer หลัก และหยุดเมื่อจุดแทรกตรวจไม่ตรงกับโค้ด ห้ามปิด assertion เพื่อให้รันผ่าน
ผลส่วนนี้วัดความตรงเชิงพิกัด ไม่ใช่คะแนนจำคำหรือการรับรองภาษามือ

## F. สิ่งที่ไม่ควรส่งขึ้น Git

ZIP ส่งต่อ, `.venv`, `models`, `SLclean.sqlite3`, น้ำหนักโมเดล, inputs/outputs ดิบ, ภาพ/วิดีโอ และ cache
กฎ `.gitignore` กันส่วนเหล่านี้ไว้แล้ว แต่ห้ามใช้ `git add -f` กับไฟล์เหล่านี้
สคริปต์วิจัยเก่าใน archive บางตัวเก็บเส้นทาง/protocol ของเครื่องเดิมไว้ จึงไม่ใช่ entry point ที่รองรับการย้ายเครื่องในชุดนี้ ให้เริ่มจากตัวเปิดและคำสั่งที่ระบุข้างต้น

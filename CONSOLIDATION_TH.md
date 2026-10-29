# รวมระบบไว้ใน newton-upload — 29 ตุลาคม 2026

## ตำแหน่งใช้งานปัจจุบัน

- เว็บ: `open_signing_avatar.cmd`
- ตัวประเมิน 200 คลิป: `run_evaluation.cmd`
- ตรวจความตรงเชิงพิกัดบน Canvas ใหม่: `run_fidelity.cmd` (สร้าง inputs ใหม่; สำรอง fidelity_results ก่อนรันซ้ำ)
- ตรวจ environment SDPose ที่เคยทำงานสำเร็จ: `run_sdpose_check.cmd`
- Python หลัก: `.venv/Scripts/python.exe`; base interpreter: `local_runtime/python312`
- RTMW/MediaPipe: `recognition_audit/rtmpose_env` และ `recognition_audit/paired_video_env` ใช้ base interpreter ภายในโปรเจกต์
- SDPose runtime ที่เลือกใช้: `local_runtime/sdpose_env` (เดิมคือ s310) พร้อม Python 3.10 ใน `evaluation/model-results/sdpose-trial-20261009/python310-official`
- Node/Chrome: `local_runtime/node` และ `local_runtime/chrome`; Vite/Playwright: `frontend-tooling/node_modules`
- โมเดลเสียง: `models/xlsr53-english`; SQLite: `SLclean.sqlite3`
- ประวัติส่งต่อ: `handoff`; ผล fidelity เดิม: `fidelity_results`
- ไฟล์เก่าที่แตกต่าง: `local_archive/tinnapat`, `local_archive/pose-animator`

## วิธีรักษาหลักฐานเก่า

ไม่แก้ JSON ผลทดลอง/protocol เก่าเพียงเพื่อเปลี่ยนเส้นทาง แต่ใช้ `portable_paths.loads` แปลงเส้นทางในหน่วยความจำตอนอ่าน
แก้สคริปต์ที่ใช้งานให้ชี้มาที่ไฟล์ภายใน newton-upload; เก็บไบต์ก่อนแก้ไว้ใน `local_archive/before-path-rebase`
หากสคริปต์ frozen ตรวจ hash ของ production ไม่ผ่าน ให้สร้าง protocol ใหม่ ไม่ปิด assertion หรืออ้างว่าเป็นผลทดลองเดิม

ไฟล์ใหญ่ใช้ hard link เมื่อทำได้: ลบชื่อไฟล์ฝั่งเก่าไม่ทำให้ไฟล์ฝั่งนี้หาย แต่ไม่ใช่สำเนาสำรองอิสระเมื่อมีการเขียนทับ
เครื่องมือเปลี่ยนเส้นทางใช้ atomic replacement เพื่อแยก hard link ก่อนแก้ ไม่เขียนทับข้อมูลต้นฉบับ
ไม่ได้ลบโฟลเดอร์ต้นฉบับใด ๆ

## หลักฐานทดสอบโดยปิดเส้นทางเดิม

ทดสอบโดยเปลี่ยนชื่อชั่วคราวของ tinnapat, signing-avatar-pose-animator, visualization model-results และ s310 แล้วคืนชื่อครบใน finally
รายงานในเครื่อง: `local_archive/isolation-20261029-210442/results.json` พร้อม log แยกรายการ

- เว็บ HTTP 200; โหลด SQLite และโมเดลเสียงจาก newton-upload; hello 179, love 105, abdomen 137 เฟรม
- Exp1 ST-GCN 122/200; Exp1 SL-GCN 96/200; Exp2 baseline 19/200; Exp3 99/200
- RTMW ตรวจภาพจริง 133 จุด ตรวจ SHA-256 น้ำหนักและภาพผ่าน
- MediaPipe ตรวจภาพ Avatar จริงได้
- SDPose: imports และ synthetic heatmap head ผ่าน (1,133,16,12) ไม่ใช่การรันน้ำหนัก SDPose ทั้งชุดหรือการยืนยันคะแนนรู้จำใหม่
- Canvas fidelity: 6 ลำดับ รวม 701 เฟรมตัวอย่าง ผ่านการจับภาพ/checksum/ตรวจ inverse camera และสรุปผล
- ชุดทดสอบหลักเดิม 28 ข้อผ่านใน isolation; เพิ่ม unit tests สำหรับ path relocation แยกต่างหาก

## ขอบเขตและข้อจำกัด

ระบบหลักและเส้นทางทดสอบข้างต้นไม่ต้องพึ่งโฟลเดอร์เดิมแล้ว แต่ไม่ได้รันทุกสคริปต์วิจัยย้อนหลังทุกเงื่อนไขใหม่
เก็บ env310/env310-official รุ่นที่ทดลองไม่สำเร็จไว้เป็นประวัติ ไม่ใช้เป็น default; ตัว env310-official เดิมมี DLL error เหมือนต้นฉบับ ส่วน runtime s310 ที่ย้ายมาใช้งานได้
ไม่มีการรับรองความถูกต้องทางภาษา ASL จากการย้ายไฟล์ และยังไม่ได้ทดสอบการพูดไมค์จริงหรือการติดตั้งทุกอย่างบน Windows เครื่องอื่น
Python environments ที่รวมครั้งนี้เป็นชุดเฉพาะเครื่องนี้; ถ้าย้ายอีกเครื่องให้ใช้ setup_friend และชุดส่งต่อที่เหมาะสม ไม่คัดลอก venv แล้วถือว่าใช้ได้ทันที

ZIP ส่งเพื่อนที่สร้างก่อนการรวมครั้งนี้ยังเป็นเวอร์ชันเดิม ไม่ใช่สำรองข้อมูลการทดลองทั้งหมดหรือ runtime ทั้งหมดของการรวมครั้งนี้
Git เก็บเฉพาะโค้ด/คู่มือ/การทดสอบ ไม่รวม `.venv`, local_runtime, local_archive, handoff, โมเดล ฐานข้อมูล ผลดิบ หรือ ZIP

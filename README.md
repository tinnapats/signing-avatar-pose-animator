# Signing Avatar / Pose Animator

เว็บแอปสำหรับแปลงข้อความหรือข้อมูลท่าทางจากไฟล์ CSV ให้เป็นอวตารภาษามือแบบเคลื่อนไหวบนเครื่องของคุณ โดยใช้ตัวเล่น [Pose Animator](https://github.com/yemount/pose-animator) ร่วมกับ Python server ขนาดเล็กสำหรับสร้างลำดับการเคลื่อนไหวจากชุดข้อมูลในเครื่อง

> **ความเป็นส่วนตัว:** โฟลเดอร์ชุดข้อมูล `SLclean/` ถูกตั้งใจไม่ให้อัปโหลดขึ้น repository นี้ ผู้ใช้แต่ละคนต้องเตรียมชุดข้อมูล CSV ของตนเองไว้ในเครื่องก่อนใช้การแปลงข้อความเป็นภาพเคลื่อนไหว

## ความสามารถ

- เล่นลำดับท่าทาง (pose animation) ผ่านเบราว์เซอร์
- สร้างลำดับท่าทางจากข้อความ โดยค้นหาไฟล์ CSV ที่ตรงกันจากชุดข้อมูลในเครื่อง
- รองรับการแปลงเสียงเป็นข้อความด้วยโมเดล XLSR-53 English (Common Voice) ในเครื่อง (ตัวเลือกเพิ่มเติม)
- เลือกใช้อวตารที่มีให้ หรือโหลดภาพประกอบ SVG ของคุณเอง

## สิ่งที่ต้องมี

- Windows และ Python 3.10 ขึ้นไป
- เบราว์เซอร์รุ่นปัจจุบัน
- หากต้องการใช้ไมโครโฟน: ติดตั้ง dependencies ใน `requirements.txt` โดยโมเดล XLSR-53 จะดาวน์โหลดครั้งแรกอัตโนมัติ

## เริ่มต้นใช้งาน

1. ดาวน์โหลดโปรเจกต์และเปิดโฟลเดอร์

   ```powershell
   git clone https://github.com/tinnapats/signing-avatar-pose-animator.git
   cd signing-avatar-pose-animator
   ```

2. สร้างและเปิดใช้งาน Python environment

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. หากต้องการใช้การรับเสียงจากไมโครโฟน ให้ติดตั้ง dependencies สำหรับ XLSR-53

   ```powershell
   pip install -r requirements.txt
   ```

4. วางชุดข้อมูล CSV ของคุณไว้ในโฟลเดอร์ `SLclean/` โฟลเดอร์นี้จะอยู่เฉพาะในเครื่องและไม่ถูกอัปโหลดขึ้น GitHub แต่ละคลิปควรเป็นไฟล์ CSV ที่ `export_pose_animator_sequence.py` อ่านได้

5. เมื่อใช้ไมโครโฟนครั้งแรก ระบบจะดาวน์โหลดโมเดล `jonatasgrosman/wav2vec2-large-xlsr-53-english` จาก Hugging Face และเก็บ cache ไว้ในเครื่อง

6. เปิด server

   ```powershell
   .\.venv\Scripts\python.exe .\run_pose_animator_server.py --port 8025
   ```

   หรือหลังจากสร้าง `.venv` แล้ว ให้ดับเบิลคลิก `open_signing_avatar.cmd`

7. เปิดลิงก์ที่แสดงในหน้าต่างคำสั่ง โดยปกติคือ

   ```text
   http://127.0.0.1:8025/dataset_player.html
   ```

## วิธีใช้งาน

- พิมพ์ข้อความในหน้าตัวเล่น แล้วเลือก **Generate From Text** เพื่อค้นหาคลิป CSV ที่ตรงกันจากชุดข้อมูลในเครื่อง
- หากมีไฟล์ลำดับท่าทาง JSON อยู่แล้ว ให้โหลดไฟล์ดังกล่าวในตัวเล่นเพื่อเล่นภาพเคลื่อนไหว
- ใช้ปุ่มควบคุมเพื่อเลือกอวตารที่มีให้ หรือโหลดไฟล์ภาพประกอบ SVG ของคุณเอง

### สร้างไฟล์ลำดับท่าทางด้วยตนเอง

```powershell
.\.venv\Scripts\python.exe .\export_pose_animator_sequence.py `
  --data-dir ".\SLclean" `
  --files "SLclean\a.csv" `
  --output ".\pose-animator\resources\data\my_sequence.json"
```

จากนั้นโหลด `pose-animator/resources/data/my_sequence.json` ในตัวเล่นบนเบราว์เซอร์

## โครงสร้างโปรเจกต์

| ตำแหน่ง | หน้าที่ |
| --- | --- |
| `run_pose_animator_server.py` | Web server ในเครื่องและ API สำหรับข้อความ/เสียง |
| `export_pose_animator_sequence.py` | แปลงคลิปท่าทาง CSV เป็น JSON สำหรับตัวเล่น |
| `pose-animator/` | ตัวเล่นบนเบราว์เซอร์และไฟล์สำหรับเรนเดอร์ |
| `SLclean/` | ชุดข้อมูลส่วนตัวในเครื่อง — ไม่ถูกเผยแพร่ |

## การรักษาความเป็นส่วนตัวของข้อมูล

ไฟล์ `.gitignore` ตั้งค่าไม่ให้อัปโหลด `SLclean/`, โมเดล Vosk ที่ดาวน์โหลด, virtual environment, ไฟล์สำรอง และไฟล์ชั่วคราวที่สร้างระหว่างทำงาน ก่อนอัปโหลดงานเพิ่ม ให้ใช้ `git status` ตรวจสอบว่าไม่มีข้อมูลส่วนตัวถูกเตรียมอัปโหลด

## เครดิต

ส่วนแสดงผลภาพเคลื่อนไหวบนเบราว์เซอร์พัฒนาต่อยอดจาก Pose Animator ดูสัญญาอนุญาตและเอกสารเพิ่มเติมได้ใน [`pose-animator/`](pose-animator/)

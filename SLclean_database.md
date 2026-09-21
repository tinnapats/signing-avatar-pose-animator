# ฐานข้อมูล SLclean

- ไฟล์ฐานข้อมูล: `C:\pro1end\SLclean.sqlite3`
- ต้นฉบับ: `C:\pro1end\SLclean` (ไม่ถูกแก้ไขหรือลบ)
- ตาราง `clips`: หนึ่งแถวต่อคลิป พร้อมคำศัพท์ พาธสัมพัทธ์ จำนวนแถว จำนวนเฟรม และข้อมูล CSV ทั้งไฟล์ใน `csv_zlib`
- ตาราง `metadata`: ข้อมูลรูปแบบและสรุปการนำเข้า
- มีดัชนี `clips_label` สำหรับค้นหาคำ/วลีโดยไม่แยกตัวพิมพ์เล็กใหญ่
- `csv_zlib` เป็น BLOB ที่บีบอัดด้วย zlib ไม่ใช่แค่ลิงก์ไปไฟล์ CSV; ข้อมูลพิกัดทุกคอลัมน์ยังอยู่ครบ
- เว็บอ่านคลิปจากฐานข้อมูลนี้เป็นค่าเริ่มต้น สามารถเปลี่ยนด้วย --data-dir
- ฐานข้อมูลเป็นสำเนา ณ เวลานำเข้า ไม่ซิงก์เมื่อ CSV เปลี่ยน

## อ่านคลิปด้วย Python

```python
import io
import sqlite3
import zlib
import pandas as pd

with sqlite3.connect(r'C:\pro1end\SLclean.sqlite3') as db:
    row = db.execute(
        'SELECT csv_zlib FROM clips WHERE label = ? COLLATE NOCASE ORDER BY id LIMIT 1',
        ('high school',),
    ).fetchone()
    if row is not None:
        clip = pd.read_csv(io.BytesIO(zlib.decompress(row[0])))
```

เครื่องมือ `import_slclean_sqlite.py` ตรวจ SHA-256 ของทุกคลิปหลังอ่านกลับจากฐานข้อมูล และตรวจความสมบูรณ์ด้วย SQLite `integrity_check` ก่อนเปลี่ยนชื่อไฟล์นำเข้าเป็นฐานข้อมูลปลายทาง เครื่องมือจะไม่เขียนทับฐานข้อมูลเดิม

ฐานข้อมูลถูกยกเว้นใน `.gitignore` เช่นเดียวกับชุดข้อมูลต้นฉบับ

# รายงาน Ablation Study ระบบมือ Signing Avatar

วันที่ทดลอง: 4 สิงหาคม 2026  
คำที่ใช้: `abdomen`, `hello`, `love`  
จำนวนการรัน: 8 configurations × 3 คำ = 24 runs  
เวลารวม: 36.66 วินาที

## วิธีทดลอง

ใช้ Full build 15 เป็นระบบอ้างอิง แล้วปิดองค์ประกอบทีละส่วน โดยใช้ CSV ต้นฉบับ พารามิเตอร์ Canvas และอัตราเฟรมชุดเดียวกันทุก configuration

ตัวชี้วัดหลัก:

- Bone span: `(P95 - P05) / median` ของความยาวกระดูก 3 มิติ ค่ายิ่งต่ำยิ่งดี
- Joint violations: สัดส่วนมุมข้อต่อที่เกิน 115 องศา
- Motion retained: สัดส่วน transition ที่นิ้วยังเคลื่อนไหว
- Shape jerk: third temporal difference ของ landmark เทียบกับขนาดฝ่ามือ ค่ายิ่งต่ำยิ่งลื่น
- Coverage: สัดส่วนเฟรมมือที่มีคะแนนอย่างน้อย 0.18
- Topology errors: คำเตือนนิ้วไขว้หรือลำดับ MCP ผิดในภาพ 2 มิติ

## ผลรวมสามคำ

| Configuration | Coverage | Bone span สูงสุด ↓ | มุมสูงสุด ↓ | Joint violations ↓ | Topology errors ↓ | Motion retained ↑ | Shape jerk ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 47.85% | 0.00% | 115.00° | 0.00% | 7.23% | 99.12% | 0.031943 |
| ไม่มี 3D SLERP | 47.85% | 0.00% | 116.16° | 0.02% | 7.23% | 98.83% | 0.048985 |
| ไม่มี Gap repair | 45.87% | 0.00% | 115.00° | 0.00% | 7.25% | 99.38% | 0.026676 |
| ไม่มี Hand smoothing | 47.85% | 0.00% | 115.00° | 0.00% | 8.96% | 95.03% | 0.071378 |
| ไม่มี Bone stabilizer | 47.85% | 0.00% | 115.00° | 0.00% | 7.23% | 98.83% | 0.032657 |
| ไม่มี Fixed-length articulation | 47.85% | 245.29% | 177.91° | 13.74% | 10.40% | 98.25% | 0.029889 |
| ไม่มี Joint limit | 47.85% | 0.00% | 175.00° | 13.74% | 7.51% | 98.54% | 0.030678 |
| Raw hand baseline | 45.40% | 243.14% | 178.20° | 14.21% | 9.45% | 94.34% | 0.065488 |

## ผล Full system รายคำ

| คำ | Coverage | Bone span | มุมสูงสุด | Motion retained | Shape jerk | Topology warnings |
|---|---:|---:|---:|---:|---:|---:|
| abdomen | 23.72% | 0.00% | 96.61° | 98.44% | 0.027796 | 4.62% |
| hello | 19.83% | 0.00% | 115.00° | 97.14% | 0.053024 | 23.94% |
| love | 100.00% | 0.00% | 115.00° | 100.00% | 0.015011 | 2.38% |

## ข้อค้นพบ

1. **Fixed-length articulation สำคัญที่สุดต่อรูปทรงมือ**  
   เมื่อปิด Bone span เพิ่มจาก 0.00% เป็น 245.29% มุมสูงสุดเพิ่มเป็น 177.91° และ Joint violations เพิ่มเป็น 13.74%

2. **Joint limit จำเป็นแม้ความยาวกระดูกจะคงที่**  
   การปิดเฉพาะ Joint limit ทำให้กระดูกยังยาวคงที่ แต่มีมุมผิดข้อกำหนด 13.74% และมุมสูงสุด 175° แสดงว่าการล็อกความยาวอย่างเดียวไม่พอ

3. **Hand smoothing ลดการสั่นอย่างชัดเจน**  
   เมื่อปิด smoothing ค่า Shape jerk เพิ่ม 123.5% และ Motion retained ลดลง 4.09 percentage points

4. **3D SLERP ช่วยช่วงพลิกฝ่ามือ**  
   เมื่อเปลี่ยนเป็น linear interpolation ค่า Shape jerk เพิ่ม 53.3% และเริ่มพบมุมเกิน 115° แม้ fixed-length articulation จะยังทำงานอยู่

5. **Gap repair เพิ่ม coverage**  
   เมื่อปิด Gap repair Coverage ลดลง 1.98 percentage points ค่า jerk ที่ดูต่ำลงเกิดจากเฟรมรอยต่อบางส่วนหายไป จึงไม่ควรตีความว่าเคลื่อนไหวดีกว่า

6. **Bone stabilizer เดิมถูก articulation ตัวใหม่ครอบหน้าที่เกือบทั้งหมด**  
   เมื่อปิด Bone stabilizer ค่า Bone span และ Joint violations ไม่เปลี่ยน และ Shape jerk เพิ่มเพียง 2.2% จึงเหมาะเป็น safety layer มากกว่าองค์ประกอบหลัก

7. **Full hybrid ดีกว่า Raw baseline ชัดเจน**  
   Raw baseline มี Bone span 243.14%, Joint violations 14.21%, Motion retained ต่ำกว่า 4.78 percentage points และ Shape jerk สูงกว่า 105.0%

## ข้อสรุป Hybrid

องค์ประกอบที่มีหลักฐานสนับสนุนให้คงไว้:

- 3D SLERP สำหรับการพลิกมือ
- Gap repair สำหรับ coverage
- Confidence-aware hand smoothing สำหรับลด jerk
- Fixed-length 3D articulation สำหรับรักษาความยาวกระดูก
- Joint-angle limit สำหรับป้องกันนิ้วหักย้อน
- Bone stabilizer เป็น safety layer ก่อน articulation

## ข้อจำกัด

- เป็น Pilot study เพียง 3 คำ ยังไม่มีนัยสำคัญทางสถิติสำหรับ Dataset ทั้งหมด
- Topology warning เป็นตัววัด 2 มิติ บางกรณีอาจเป็นการซ้อนนิ้วตาม perspective จริง
- `hello` ยังมี Topology warnings 23.94% จึงควรตรวจเฟรมพลิกมือด้วยสายตาและแยก true error ออกจาก perspective overlap
- งานฉบับรายงานควรเพิ่มจำนวนคำ ใช้ bootstrap confidence interval และ paired statistical test รายคลิป

## ไฟล์ผลลัพธ์

- `hand_ablation_per_sign.csv` — ผลทุก configuration แยกรายคำ
- `hand_ablation_results.json` — ผลดิบและ metadata
- `hand_ablation_report.md` — รายงานภาษาอังกฤษที่สร้างอัตโนมัติ
- `hand_ablation_report_th.md` — รายงานภาษาไทยฉบับนี้
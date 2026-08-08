# รายงานต่อยอดการจูนมือ Hybrid — Build 16

วันที่ทดลอง: 4 สิงหาคม 2026  
คำที่ใช้: `abdomen`, `hello`, `love`, `accept`, `across`, `airplane`, `alphabet`, `angry`, `animal`, `answer`  
จำนวนการทดลอง: 9 configurations × 10 คำ = 90 runs  
เวลารวม: 155.05 วินาที

## ค่าที่ชนะ

- Hand smoothing sigma: `1.2`
- Hand smoothing radius: `2`
- Joint-angle limit: `115°`
- 3D SLERP, gap repair, bone stabilization และ fixed-length articulation ยังคงเปิดทั้งหมด

## ผลเทียบกับค่าเดิม

| ตัวชี้วัด | ค่าเดิม 0.8 / radius 1 | ค่าใหม่ 1.2 / radius 2 | ผลต่าง |
|---|---:|---:|---:|
| Shape jerk ↓ | 0.040595 | 0.024665 | ลดลง 39.2% |
| Motion retained ↑ | 97.44% | 98.93% | เพิ่มขึ้น 1.49 จุดเปอร์เซ็นต์ |
| Bone span ↓ | 0.00% | 0.00% | กระดูกยังคงความยาว |
| Joint violations ↓ | 0.01% | 0.00% | ไม่พบมุมเกินเกณฑ์ |
| Topology warning ↓ | 27.18% | 27.04% | ลดลงเล็กน้อย |
| Coverage | 58.17% | 58.17% | ไม่เสีย coverage |

## ข้อค้นพบเพิ่มเติม

1. การเพิ่ม smoothing radius จาก 1 เป็น 2 ให้ผลมากกว่าการเพิ่ม sigma โดยยังใช้ radius 1
2. การลด Joint limit เหลือ 110° ไม่ได้ลด jerk เพิ่มอย่างมีนัยสำคัญ และเพิ่ม Topology warning จึงคง 115°
3. การลด Joint limit เหลือ 105° ทำให้ Topology warning สูงขึ้นเป็น 30.28% จึงเข้มเกินไปสำหรับข้อมูลชุดนี้
4. ค่าใหม่ทำให้การเคลื่อนไหวลื่นขึ้นโดยไม่ต้องแลกกับความยาวกระดูก การเคลื่อนไหวนิ้ว หรือ coverage

## การนำไปใช้

ค่าที่ชนะถูกตั้งเป็นค่าเริ่มต้นของ `build 16` แล้ว พร้อม metadata `handTuningProfile = smooth-s12-r2-a115`

ควรตรวจภาพจริงเพิ่มเติม โดยเน้น `hello`, `alphabet`, `angry` และเฟรมที่นิ้วซ้อนกัน เพราะ Topology warning เป็นการตรวจในภาพ 2 มิติและอาจนับ perspective overlap เป็นข้อผิดพลาดได้

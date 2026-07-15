import sys

with open('D:/project_1/test (2).py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "status.object =" in line and i + 1 < len(lines) and "ไม่พบไฟล์ที่ตรงกับคำ:" in lines[i+1]:
        lines[i+1] = lines[i+1].replace("ไม่พบไฟล์ที่ตรงกับคำ:", "❌ **ไม่พบไฟล์ที่ตรงกับคำ:** `")
        lines[i+1] = lines[i+1].replace("\\n\"", "`\\n\"")
    elif "ไฟล์ทั้งหมด:" in line and "status.object" not in line:
        lines[i] = lines[i].replace("ไฟล์ทั้งหมด:", "📁 ไฟล์ทั้งหมด:")
    elif "status.object = 'กำลังสร้างแอนิเมชัน...'" in line:
        lines[i] = line.replace("'กำลังสร้างแอนิเมชัน...'", "'⏳ **กำลังสร้างแอนิเมชัน...**'")
    elif "status.object = f\"ไม่พบคำ:" in line:
        lines[i] = line.replace("ไม่พบคำ:", "⚠️ **ไม่พบคำ:** `")
        lines[i] = line.replace("} | เล่นแอนิเมชันแล้ว\"", "}` | ▶️ เล่นแอนิเมชันบนไฟล์ที่พบแล้ว\"")
    elif "status.object = 'เล่นแอนิเมชันแล้ว'" in line:
        lines[i] = line.replace("'เล่นแอนิเมชันแล้ว'", "'✅ **สร้างและเล่นแอนิเมชันสำเร็จแล้ว**'")

with open('D:/project_1/test (2).py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("Updated lines")

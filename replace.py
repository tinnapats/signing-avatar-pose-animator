import re

with open('D:/project_1/test (2).py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """        if not keys:
            missing_text = ' '.join(missing_terms) if missing_terms else '(ว่าง)'
            status.object = (
                f"ไม่พบไฟล์ที่ตรงกับคำ: {missing_text}\\n"
                f"ไฟล์ทั้งหมด: {len(CLIPS)}"
            )
            return

        # เน เธชเธ”เธ‡เธชเธ–เธฒเธ™เธฐ "เธ เธณเธฅเธฑเธ‡เธชเธฃเน‰เธฒเธ‡ animation"
        status.object = 'กำลังสร้างแอนิเมชัน...'
        
        new_df = build_sequence_from_tokens(keys)
        set_sequence(new_df)

        if missing_terms:
            status.object = f"ไม่พบคำ: {' '.join(missing_terms)} | เล่นแอนิเมชันแล้ว"
        else:
            status.object = 'เล่นแอนิเมชันแล้ว'"""

replacement = """        if not keys:
            missing_text = ' '.join(missing_terms) if missing_terms else '(ว่าง)'
            status.object = (
                f"❌ **ไม่พบไฟล์ที่ตรงกับคำ:** `{missing_text}`\\n"
                f"📁 ไฟล์ทั้งหมด: {len(CLIPS)}"
            )
            return

        # เน เธชเธ”เธ‡เธชเธ–เธฒเธ™เธฐ "เธ เธณเธฅเธฑเธ‡เธชเธฃเน‰เธฒเธ‡ animation"
        status.object = '⏳ **กำลังสร้างแอนิเมชัน...**'
        
        new_df = build_sequence_from_tokens(keys)
        set_sequence(new_df)

        if missing_terms:
            status.object = f"⚠️ **ไม่พบคำ:** `{' '.join(missing_terms)}` | ▶️ เล่นแอนิเมชันแล้ว"
        else:
            status.object = '✅ **สร้างและเล่นแอนิเมชันสำเร็จแล้ว**'"""

# Normalize CR LF to LF for comparison just in case
text_norm = text.replace('\\r\\n', '\\n')
if target in text_norm:
    new_text = text_norm.replace(target, replacement)
    with open('D:/project_1/test (2).py', 'w', encoding='utf-8', newline='') as f:
        f.write(new_text)
    print("Replaced successfully")
else:
    print("Target not found")

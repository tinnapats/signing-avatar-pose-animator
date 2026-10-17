import { apiUrl } from './api.js';

export function installComparison(capture) {
  if (new URLSearchParams(location.search).has('previewWord')) return;
  const panel = document.createElement('section');
  panel.className = 'avatar-output-card';
  panel.style.cssText = 'margin-top:16px;padding:20px';
  panel.innerHTML = `<h2 style="font-size:16px;margin-top:0">เปรียบเทียบท่าทางกับฐานข้อมูล <small>ทดลอง</small></h2>
    <p>Generate คำหรือวลีเดียว เช่น you หรือ high school แล้วตรวจภาพอวตาร 12 เฟรมด้วย AI</p>
    <button id="comparePoseBtn" type="button">ตรวจด้วย MediaPipe</button>
    <p id="comparisonMessage" role="status" aria-live="polite"></p>
    <div id="comparisonResults"></div>`;
  document.querySelector('.viewer').append(panel);
  const button = panel.querySelector('button');
  const message = panel.querySelector('#comparisonMessage');
  const results = panel.querySelector('#comparisonResults');
  button.onclick = async () => {
    button.disabled = true;
    results.replaceChildren();
    message.textContent = 'กำลังเก็บภาพและตรวจด้วย AI…';
    try {
      const captured = capture();
      const response = await fetch(apiUrl('/api/compare_pose'), {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(captured), signal: AbortSignal.timeout(180000),
      });
      const report = await response.json();
      if (!response.ok || !report.ok) throw new Error(report.error || 'ตรวจไม่สำเร็จ');
      message.textContent = `ผลของ ${report.clip} • ${captured.avatar} • ตรวจ ${report.sampleCount} เฟรม`;
      const scores = report.summary.similarity;
      const overall = document.createElement('p');
      overall.style.cssText = 'font-size:24px;font-weight:600;margin:16px 0 8px';
      overall.textContent = `ความเหมือนรวม: ${scores.overall == null ? 'ข้อมูลไม่ครบ' : scores.overall.toFixed(1) + '%'}`;
      const coverage = document.createElement('p');
      coverage.textContent = `คำนวณจาก ${report.summary.overallFrames}/${report.sampleCount} เฟรมที่เทียบได้ครบทั้งแขน ข้อมือ และนิ้ว`;
      results.append(overall, coverage);
      const metrics = [
        ['ความเหมือนของมุมข้อศอก', 'elbowDegrees', '°'],
        ['ความเหมือนของตำแหน่งข้อมือ', 'wristDistance', ' เท่าของความกว้างไหล่'],
        ['ความเหมือนของมุมข้อนิ้ว', 'fingerDegrees', '°'],
      ];
      for (const [label, key, unit] of metrics) {
        const p = document.createElement('p');
        const value = report.summary[key];
        p.textContent = `${label}: ${scores[key] == null ? 'ตรวจไม่ได้' : scores[key].toFixed(1) + '%'} — เทียบได้ ${report.summary[key + 'Frames']}/${report.sampleCount} เฟรม${value == null ? '' : ` (ต่างเฉลี่ย ${value.toFixed(2)}${unit})`}`;
        results.append(p);
      }
      const note = document.createElement('p');
      note.className = 'hint';
      note.textContent = 'เปอร์เซ็นต์ยิ่งสูงยิ่งใกล้ต้นฉบับตามสูตรทดลอง ไม่ใช่ความมั่นใจของ AI หรือความถูกต้องของภาษามือ • คะแนนรวมใช้เฉพาะเฟรมที่มีข้อมูลครบ จึงอาจไม่แทนทั้งคลิป • จับคู่เวลาโดยประมาณ ยังไม่ใช้ DTW • AI อาจอ่านภาพการ์ตูนผิด';
      results.append(note);
      const formula = document.createElement('details');
      const formulaTitle = document.createElement('summary');
      formulaTitle.textContent = 'วิธีคิดเปอร์เซ็นต์';
      const formulaText = document.createElement('p');
      formulaText.textContent = 'มุมแขน/นิ้ว: 100 × (1 − มุมที่ต่าง ÷ 180) โดยจำกัด 0–100% • ข้อมือ: 100 ÷ (1 + ระยะที่ต่างในหน่วยความกว้างไหล่) • คะแนนรวม: เฉลี่ยสามส่วนด้วยน้ำหนักเท่ากันในแต่ละเฟรมที่มีครบ แล้วเฉลี่ยเฟรมเหล่านั้น ไม่เติมคะแนนให้จุดที่ตรวจไม่พบ และยังไม่มีเกณฑ์ผ่าน/ไม่ผ่าน';
      formula.append(formulaTitle, formulaText); results.append(formula);
      const select = document.createElement('select');
      select.setAttribute('aria-label', 'เลือกเฟรมเปรียบเทียบ');
      report.frames.forEach((f, i) => select.add(new Option(`เฟรม Output ${f.outputFrame} / ต้นฉบับ ${f.sourceFrame}`, String(i))));
      const visual = document.createElement('div');
      visual.style.cssText = 'display:flex;flex-wrap:wrap;gap:16px;margin-top:12px';
      const img = document.createElement('img');
      img.alt = 'ภาพอวตารที่ส่งให้ AI ตรวจ';
      img.style.cssText = 'width:280px;max-width:100%;height:auto;object-fit:contain;border:1px solid #ddd';
      const overlay = document.createElement('canvas');
      overlay.width = 360; overlay.height = 360;
      overlay.style.cssText = 'width:360px;max-width:100%;border:1px solid #ddd';
      overlay.setAttribute('aria-label', 'โครงแขนต้นฉบับสีน้ำเงินและ AI สีส้ม');
      const detail = document.createElement('p');
      const draw = () => {
        const i = Number(select.value), f = report.frames[i];
        img.src = captured.samples[i].image;
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, 360, 360);
        const points = [...Object.values(f.reference), ...Object.values(f.detected)];
        const extent = Math.max(1.5, ...points.flatMap(p => [Math.abs(p[0]), Math.abs(p[1])]));
        const xy = p => [180 + p[0] * 145 / extent, 180 + p[1] * 145 / extent];
        for (const [key, color] of [['reference', '#2367b1'], ['detected', '#c66b14']]) {
          ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
          for (const [a, b] of [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]) {
            if (!f[key][a] || !f[key][b]) continue;
            ctx.beginPath(); ctx.moveTo(...xy(f[key][a])); ctx.lineTo(...xy(f[key][b])); ctx.stroke();
          }
          for (const p of Object.values(f[key])) { ctx.beginPath(); ctx.arc(...xy(p), 4, 0, 2 * Math.PI); ctx.fill(); }
        }
        ctx.fillStyle = '#2367b1'; ctx.fillText('Source (blue)', 12, 20);
        ctx.fillStyle = '#c66b14'; ctx.fillText('AI (orange)', 12, 38);
        detail.textContent = `ต้นฉบับมีจุดแขนที่เชื่อถือได้ ${f.sourceBodyPoints}/6 จุด • AI พบจุดแขน ${f.detectedBodyPoints}/6 จุด • AI พบมือ ${f.detectedHands} ข้าง • ต้นฉบับมีมือ ${f.sourceHands} ข้าง • จับคู่มือได้ ${f.pairedHands} ข้าง`;
      };
      select.onchange = draw;
      visual.append(img, overlay); results.append(select, visual, detail);
      const download = document.createElement('button');
      download.textContent = 'ดาวน์โหลดผล JSON';
      download.onclick = () => {
        const url = URL.createObjectURL(new Blob([JSON.stringify({...report, avatar: captured.avatar}, null, 2)], {type:'application/json'}));
        const link = document.createElement('a'); link.href = url; link.download = 'pose-comparison.json'; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      };
      results.append(download); draw();
    } catch (error) {
      message.textContent = `ตรวจไม่ได้: ${error.message}`;
    } finally { button.disabled = false; }
  };
}

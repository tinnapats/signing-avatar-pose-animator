const params = new URLSearchParams(location.search);
const previewWord = params.get('previewWord');
if (previewWord) document.body.classList.add('word-preview-page');
const styles = document.createElement('style');
styles.textContent = `
.output-word { border:0; border-bottom:1px dotted #888; border-radius:0; padding:2px 3px; min-height:32px; background:transparent; font:inherit; color:inherit; cursor:pointer; }
.output-word:hover,.output-word:focus-visible { background:#eee; border-bottom-color:#222; }
.word-popup { position:fixed; z-index:100; width:320px; max-width:calc(100vw - 16px); background:white; border:1px solid #bbb; border-radius:6px; box-shadow:0 6px 24px #0002; overflow:hidden; }
.word-popup[hidden] { display:none; }
.word-popup-header { display:flex; align-items:center; justify-content:space-between; padding:6px 12px; border-bottom:1px solid #ddd; font-size:14px; }
.word-popup-close { border:0; background:transparent; min-height:28px; padding:2px 8px; font-size:20px; }
.word-popup iframe { display:block; width:100%; height:320px; border:0; }
.word-preview-page { background:white; overflow:hidden; }
.word-preview-page .topbar,.word-preview-page .panel,.word-preview-page .details,.word-preview-page .output-area { display:none; }
.word-preview-page .sidebar { display:contents; }
.word-preview-page .layout { display:block; min-height:0; padding:0; margin:0; }
.word-preview-page .viewer { padding:0; }
.word-preview-page .avatar-output-card { border:0; margin:0; }
.word-preview-page .canvas-wrap { width:100%; height:320px; }
.word-preview-page .status { position:absolute; bottom:0; left:0; width:100%; margin:0; padding:4px 8px; font-size:11px; background:#fffd; }
`;
document.head.appendChild(styles);
let popup, frame, caption, opener, showTimer, hideTimer;
function closePopup() {
  clearTimeout(showTimer); clearTimeout(hideTimer);
  if (!popup) return;
  popup.hidden = true;
  if (opener) opener.setAttribute('aria-expanded', 'false');
  frame.removeAttribute('src');
  opener = null;
}
function scheduleClose() { clearTimeout(hideTimer); hideTimer = setTimeout(closePopup, 180); }
function ensurePopup() {
  if (popup) return;
  popup = document.createElement('div');
  popup.className = 'word-popup'; popup.id = 'wordPreviewPopup'; popup.hidden = true;
  popup.setAttribute('role', 'dialog'); popup.setAttribute('aria-label', 'ตัวอย่างท่าทาง');
  const header = document.createElement('div'); header.className = 'word-popup-header';
  caption = document.createElement('strong');
  const close = document.createElement('button'); close.className = 'word-popup-close';
  close.type = 'button'; close.textContent = '×'; close.setAttribute('aria-label', 'ปิดตัวอย่าง');
  close.onclick = () => { const previous = opener; closePopup(); if (previous) { previous.focus(); clearTimeout(showTimer); } };
  header.append(caption, close);
  frame = document.createElement('iframe'); frame.title = 'ตัวอย่างท่าทางของคำ';
  popup.append(header, frame); document.body.appendChild(popup);
  popup.addEventListener('mouseenter', () => clearTimeout(hideTimer));
  popup.addEventListener('mouseleave', scheduleClose);
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closePopup(); });
  document.addEventListener('pointerdown', (event) => {
    if (!popup.hidden && !popup.contains(event.target) && event.target !== opener) closePopup();
  });
  window.addEventListener('resize', closePopup);
  window.addEventListener('scroll', closePopup, true);
}
function openPopup(button, word) {
  clearTimeout(hideTimer); clearTimeout(showTimer);
  ensurePopup();
  if (opener === button && !popup.hidden) return;
  closePopup(); opener = button; button.setAttribute('aria-expanded', 'true');
  caption.textContent = word; popup.hidden = false;
  const rect = button.getBoundingClientRect();
  const width = popup.offsetWidth, height = popup.offsetHeight;
  popup.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - width - 8))}px`;
  popup.style.top = `${Math.max(8, rect.top >= height + 8 ? rect.top - height - 8 : Math.min(rect.bottom + 8, innerHeight - height - 8))}px`;
  const url = new URL(location.href); url.searchParams.set('previewWord', word);
  url.searchParams.set('previewAvatar', document.getElementById('avatarSelect').value);
  frame.src = url.href;
}
export function renderWordOutput(words) {
  closePopup();
  const output = document.getElementById('wordOutput'); output.replaceChildren();
  if (!words.length || previewWord) { output.textContent = words.join(' → ') || '—'; return; }
  words.forEach((word, index) => {
    if (index) output.appendChild(document.createTextNode(' → '));
    const button = document.createElement('button');
    button.className = 'output-word'; button.type = 'button'; button.textContent = word;
    button.setAttribute('aria-label', `ดูท่าทาง ${word}`); button.setAttribute('aria-haspopup', 'dialog');
    button.setAttribute('aria-expanded', 'false'); button.setAttribute('aria-controls', 'wordPreviewPopup');
    const schedule = () => { clearTimeout(hideTimer); clearTimeout(showTimer); showTimer = setTimeout(() => openPopup(button, word), 200); };
    button.addEventListener('mouseenter', schedule); button.addEventListener('focus', schedule);
    button.addEventListener('mouseleave', () => { clearTimeout(showTimer); scheduleClose(); });
    button.addEventListener('blur', () => { clearTimeout(showTimer); scheduleClose(); });
    button.addEventListener('click', () => openPopup(button, word));
    output.appendChild(button);
  });
}
export async function startWordPreview(generate, loadAvatar, avatars) {
  if (!previewWord) return;
  const avatar = params.get('previewAvatar');
  if (avatar && avatars[avatar]) await loadAvatar(avatar);
  document.getElementById('textPrompt').value = previewWord;
  document.getElementById('loopCheck').checked = true;
  await generate();
  const status = document.getElementById('status');
  if (status.textContent.startsWith('Loaded ')) status.textContent = '';
}

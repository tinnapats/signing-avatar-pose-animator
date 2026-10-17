const base = (import.meta.env.VITE_API_BASE_URL || '').trim().replace(/\/+$/, '');
if (base && !/^https?:\/\/[^/]+(?:\/[^?#]*)?$/.test(base)) {
  throw new Error('VITE_API_BASE_URL must be an HTTP(S) server URL without query parameters.');
}
export function apiUrl(path) {
  return `${base}${path}`;
}

export function showConnectionState(message) {
  let notice = document.getElementById('connectionNotice');
  if (!notice) {
    notice = document.createElement('p');
    notice.id = 'connectionNotice';
    notice.className = 'hint';
    notice.setAttribute('role', 'status');
    document.querySelector('.input-panel')?.append(notice);
  }
  notice.textContent = message;
  notice.hidden = !message;
}

import { API_BASE } from '../configs/labels';

async function getJson(url, opts, timeoutMs) {
  const controller = new AbortController();
  const timer = timeoutMs ? window.setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const res = await fetch(API_BASE + url, { ...(opts || {}), signal: controller.signal });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.error || `HTTP ${res.status}`);
    return data;
  } catch (e) {
    if (e && e.name === 'AbortError') {
      throw new Error('Yêu cầu quá thời gian chờ. Vui lòng thử lại.');
    }
    throw e;
  } finally {
    if (timer) window.clearTimeout(timer);
  }
}

export const fetchHealth = () => getJson('/api/health', {}, 10000);
// Predict: tối đa 120s (Grad-CAM có thể lâu)
export const predictImage = (formData) => getJson('/api/predict', { method: 'POST', body: formData }, 120000);

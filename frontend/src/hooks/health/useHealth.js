import { useState, useCallback, useEffect, useRef } from 'react';
import { fetchHealth } from '../../api';

// Poll fast (5s) when disconnected, slow (30s) when connected
const RETRY_INTERVAL_DISCONNECTED = 5000;
const RETRY_INTERVAL_CONNECTED = 30000;

export function useHealth(toast) {
  const [health, setHealth] = useState({
    loading: true,
    text: "Đang kiểm tra...",
    cnn: false,
    maxUploadMb: 10,
    modelVersion: "",
    gpu: "",
  });

  // Track previous cnn state to detect transitions
  const prevCnn = useRef(null);
  const timerRef = useRef(null);

  const refreshHealth = useCallback(async () => {
    try {
      const data = await fetchHealth();
      const cnnOk = !!data.cnn_loaded;
      setHealth({
        loading: false,
        text: cnnOk ? "CNN sẵn sàng" : "CNN chưa tải",
        cnn: cnnOk,
        maxUploadMb: Number(data.max_upload_mb) > 0 ? Number(data.max_upload_mb) : 10,
        modelVersion: data.model_version || "",
        gpu: data.gpu || "",
      });
      // Toast only on transition: disconnected → connected
      if (cnnOk && prevCnn.current === false) {
        toast("Hệ thống sẵn sàng", "success");
      } else if (cnnOk && prevCnn.current === null) {
        // First successful load
        toast("Hệ thống sẵn sàng", "success");
      }
      prevCnn.current = cnnOk;
      return cnnOk;
    } catch {
      setHealth(prev => ({
        ...prev,
        loading: false,
        text: "Không kết nối được máy chủ",
        cnn: false,
      }));
      // Toast only on transition: connected → disconnected
      if (prevCnn.current === true) {
        toast("Mất kết nối với máy chủ", "error");
      }
      prevCnn.current = false;
      return false;
    }
  }, [toast]);

  // Auto-polling: fast when disconnected, slow when connected
  useEffect(() => {
    let cancelled = false;

    async function poll() {
      if (cancelled) return;
      const ok = await refreshHealth();
      if (cancelled) return;
      const delay = ok ? RETRY_INTERVAL_CONNECTED : RETRY_INTERVAL_DISCONNECTED;
      timerRef.current = window.setTimeout(poll, delay);
    }

    poll();

    return () => {
      cancelled = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { health, refreshHealth };
}

import { useState, useCallback } from 'react';
import { predictImage } from '../../api';

export function usePredict(toast) {
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);

  const runPredict = useCallback(async ({
    file, threshold, enableGradcam, viewPosition, onSuccess,
  }) => {
    if (!file || predicting) return;
    setPredicting(true);
    const fd = new FormData();
    fd.append("image", file);
    fd.append("threshold", String(threshold));
    fd.append("enable_gradcam", enableGradcam ? "true" : "false");
    fd.append("view_position", viewPosition);
    try {
      const data = await predictImage(fd);
      setResult(data);
      if (typeof data.image_resolution_note === "string" && data.image_resolution_note.trim()) {
        toast(data.image_resolution_note, "info");
      }
      toast("Phân tích hoàn tất", "success");
      if (onSuccess) onSuccess(data);
    } catch (e) {
      toast(`Lỗi: ${e.message}`, "error");
    } finally {
      setPredicting(false);
    }
  }, [predicting, toast]);

  const resetResult = useCallback(() => setResult(null), []);

  return { predicting, result, runPredict, resetResult };
}

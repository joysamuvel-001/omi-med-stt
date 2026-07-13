/**
 * hooks/useRecorder.js
 * ---------------------
 * Encapsulates MediaRecorder lifecycle.
 * Returns controls and state; never touches the UI.
 *
 * Usage:
 *   const { recording, elapsed, start, stop } = useRecorder(onStop);
 *
 *   onStop(blob) is called with the complete audio Blob when recording ends.
 */

import { useState, useRef, useEffect, useCallback } from "react";

export function useRecorder(onStop) {
  const [recording, setRecording]   = useState(false);
  const [elapsed,   setElapsed]     = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef        = useRef([]);
  const timerRef         = useRef(null);

  // Tick elapsed seconds while recording
  useEffect(() => {
    if (recording) {
      timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);
    } else {
      clearInterval(timerRef.current);
      setElapsed(0);
    }
    return () => clearInterval(timerRef.current);
  }, [recording]);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunksRef.current = [];

      const mr = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
      mr.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data);
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        onStop(blob);
      };

      mr.start(200);
      mediaRecorderRef.current = mr;
      setRecording(true);
    } catch {
      throw new Error("Microphone access denied. Check browser permissions and try again.");
    }
  }, [onStop]);

  const stop = useCallback(() => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }, []);

  const cancel = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      const mr = mediaRecorderRef.current;
      mr.onstop = () => {
        if (mr.stream) {
          mr.stream.getTracks().forEach((t) => t.stop());
        }
      };
      mr.stop();
    }
    setRecording(false);
  }, []);

  return { recording, elapsed, start, stop, cancel };
}
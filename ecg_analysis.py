# ecg_analysis.py
import numpy as np
from scipy.signal import find_peaks

class ECGAnalyzer:
    def __init__(self, fs=250, min_bpm=40, max_bpm=200):
        self.fs = fs
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def detect_r_peaks(self, signal):
        """
        كشف قمم R بشكل بسيط.
        distance: أقل مسافة بين قمّتين بناءً على max_bpm
        prominence: لتقليل القمم الوهمية بسبب الضوضاء
        """
        x = np.asarray(signal, dtype=float)

        # أقل مسافة بين النبضات (بالعينات)
        min_distance = int((60 / self.max_bpm) * self.fs)  # samples

        # prominence ديناميكي حسب تذبذب الإشارة
        prom = 0.5 * np.std(x)

        peaks, props = find_peaks(
            x,
            distance=max(1, min_distance),
            prominence=prom
        )
        return peaks, props

    def rr_intervals(self, peaks):
        """RR intervals بالثواني"""
        peaks = np.asarray(peaks, dtype=int)
        if len(peaks) < 2:
            return np.array([])
        return np.diff(peaks) / self.fs

    def mean_rr(self, rr):
        return float(np.mean(rr)) if len(rr) else float("nan")

    def bpm(self, rr):
        """BPM من متوسط RR"""
        if len(rr) == 0:
            return float("nan")
        return float(60.0 / np.mean(rr))

    def signal_stats(self, signal):
        x = np.asarray(signal, dtype=float)
        return {
            "max": float(np.max(x)),
            "min": float(np.min(x)),
            "std": float(np.std(x)),
        }

    def analyze(self, filtered_signal):
        """يرجع كل النتائج مرة واحدة"""
        peaks, _ = self.detect_r_peaks(filtered_signal)
        rr = self.rr_intervals(peaks)

        return {
            "peaks": peaks,
            "rr": rr,
            "mean_rr": self.mean_rr(rr),
            "bpm": self.bpm(rr),
            **self.signal_stats(filtered_signal)
        }

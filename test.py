# main.py
import numpy as np
import matplotlib.pyplot as plt

from ecg_filter import ECGFilter
from ecg_analysis import ECGAnalyzer

def main():
    fs = 250
    duration = 10  # ثواني
    t = np.linspace(0, duration, fs * duration, endpoint=False)

    # إشارة تجريبية (مش ECG حقيقي، بس للتجربة)
    # 1.2 Hz ≈ 72 BPM + Noise
    raw = np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.random.randn(len(t))

    # 1) فلترة
    ecg_filter = ECGFilter(fs=fs, lowcut=0.5, highcut=40.0, order=5)
    filtered = ecg_filter.bandpass_filter(raw)

    # 2) تحليل
    analyzer = ECGAnalyzer(fs=fs, min_bpm=40, max_bpm=200)
    results = analyzer.analyze(filtered)

    print("===== ECG Analysis نتائج =====")
    print("Mean RR (s):", results["mean_rr"])
    print("BPM:", results["bpm"])
    print("Max:", results["max"])
    print("Min:", results["min"])
    print("STD:", results["std"])
    print("Detected peaks:", len(results["peaks"]))

    # 3) رسم
    peaks = results["peaks"]
    plt.figure(figsize=(12, 5))
    plt.plot(t, filtered, label="Filtered ECG")
    if len(peaks) > 0:
        plt.plot(t[peaks], filtered[peaks], "o", label="R-peaks")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("ECG Filter + Analysis")
    plt.show()

if __name__ == "__main__":
    main()

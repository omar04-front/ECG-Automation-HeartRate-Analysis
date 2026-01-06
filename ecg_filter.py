import json
import numpy as np
import pyodbc
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


# =========================
# 1) DB Client (SQL Server)
# =========================
class SQLServerECGRepository:
    def __init__(self, server, database, trusted_connection=True, username=None, password=None, driver="ODBC Driver 17 for SQL Server"):
        self.server = server
        self.database = database
        self.trusted = trusted_connection
        self.username = username
        self.password = password
        self.driver = driver

    def _conn_str(self):
        if self.trusted:
            return (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                "Trusted_Connection=yes;"
            )
        else:
            return (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                "TrustServerCertificate=yes;"
            )

    def fetch_raw_ecg_json(self, ecg_id: int, user_id: int):
        """
        يسحب raw_ecg_signal من جدول ECG_Signal بشرط ECG_id و user_id
        ويرجعه كـ (signal_array, fs)
        """
        conn = pyodbc.connect(self._conn_str())
        cur = conn.cursor()

        cur.execute("""
            SELECT raw_ecg_signal
            FROM ECG_Signal
            WHERE ECG_id = ? AND user_id = ?
        """, (ecg_id, user_id))

        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            raise ValueError("ECG not found OR not authorized for this user_id")

        raw = row[0]

        # raw ممكن يبقى bytes (VARBINARY) أو str (NVARCHAR)
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")

        obj = json.loads(raw)

        # شكل JSON الأفضل: {"fs":250, "samples":[...]}
        if isinstance(obj, dict) and "samples" in obj:
            fs = int(obj.get("fs", 250))
            samples = obj["samples"]
            signal = np.array(samples, dtype=np.float32)
            return signal, fs

        # لو JSON قائمة مباشرة: [ ... ]
        if isinstance(obj, list):
            signal = np.array(obj, dtype=np.float32)
            return signal, 250

        raise ValueError("JSON format not recognized. Expected list or {samples:...}.")


# =========================
# 2) ECG Filter (OOP)
# =========================
class ECGFilter:
    def __init__(self, fs=250, lowcut=0.5, highcut=40.0, order=5):
        """
        fs : معدل العينة (Hz)
        lowcut : أقل تردد يمر (Hz)
        highcut : أعلى تردد يمر (Hz)
        order : رتبة الفلتر
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """تطبيق فلتر Bandpass على إشارة ECG"""
        signal = np.asarray(signal, dtype=np.float32)

        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq

        if not (0 < low < high < 1):
            raise ValueError("lowcut/highcut غير مناسبين بالنسبة لـ fs")

        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, signal)


# =========================
# 3) تشغيل: DB -> Filter -> Plot
# =========================
if __name__ == "__main__":
    SERVER_NAME = r"OMAR19\SQLEXPRESS01"         
    DB_NAME = "test"
    ECG_ID = 1
    USER_ID = 1
    # -------------------

    repo = SQLServerECGRepository(server=SERVER_NAME, database=DB_NAME, trusted_connection=True)

    # 1) سحب الإشارة من الداتابيز
    ecg_signal, fs = repo.fetch_raw_ecg_json(ecg_id=ECG_ID, user_id=USER_ID)

    # 2) إنشاء محور الزمن
    t = np.arange(len(ecg_signal)) / fs

    # 3) فلترة
    ecg_filter = ECGFilter(fs=fs, lowcut=0.5, highcut=40.0, order=5)
    filtered_ecg = ecg_filter.bandpass_filter(ecg_signal)

    # 4) رسم
    plt.figure(figsize=(10, 5))
    plt.plot(t, ecg_signal, label="Raw ECG", alpha=0.6)
    plt.plot(t, filtered_ecg, label="Filtered ECG", linewidth=2)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"ECG Filtering from DB (ECG_ID={ECG_ID}, USER_ID={USER_ID})")
    plt.show()

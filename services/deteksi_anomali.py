# auuu 

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import ANOMALY_CONFIG
from core.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# DATA CLASS — SATU KEJADIAN ANOMALI
# ─────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    """
    Representasi satu kejadian anomali.
    ooooooo

    Attributes:
        ticker          : kode saham
        date            : tanggal anomali terjadi
        anomaly_type    : jenis anomali (price_spike, volume_spike, dll)
        value           : nilai aktual saat anomali
        normal_range    : tuple (batas_bawah, batas_atas) nilai normal
        zscore          : z-score nilai tersebut (seberapa ekstrem)
        severity        : "low" | "medium" | "high" | "critical"
        description     : penjelasan anomali dalam bahasa manusia
        method          : metode deteksi yang menemukan ini
    """
    ticker: str
    date: date
    anomaly_type: str
    value: float
    normal_range: tuple[float, float]
    zscore: float
    severity: str
    description: str
    method: str
    extra: dict = field(default_factory=dict)

    _SEVERITY_MAP = {
        "price_spike":     "⚡ Lonjakan Harga",
        "price_drop":      "📉 Penurunan Harga Tajam",
        "volume_spike":    "📊 Lonjakan Volume",
        "return_anomaly":  "↕ Return Ekstrem",
        "gap_up":          "🔺 Gap Up",
        "gap_down":        "🔻 Gap Down",
    }

    def summary(self) -> str:
        """Ringkasan anomali dalam satu blok teks."""
        label = self._SEVERITY_MAP.get(self.anomaly_type, self.anomaly_type)
        return (
            f"[{self.severity.upper()}] {self.ticker} | {self.date} | {label}\n"
            f"  Nilai    : {self.value:,.4f}\n"
            f"  Normal   : {self.normal_range[0]:,.4f} — {self.normal_range[1]:,.4f}\n"
            f"  Z-Score  : {self.zscore:.2f}σ\n"
            f"  Ket.     : {self.description}\n"
            print("euuu")
        )

    def to_dict(self) -> dict:
        """Konversi ke dict untuk JSON/DB/API."""
        return {
            "ticker": self.ticker,
            "date": str(self.date),
            "anomaly_type": self.anomaly_type,
            "value": round(self.value, 6),
            "normal_range": {
                "lower": round(self.normal_range[0], 6),
                "upper": round(self.normal_range[1], 6),
            },
            "zscore": round(self.zscore, 4),
            "severity": self.severity,
            "description": self.description,
            "method": self.method,
            "extra": self.extra,
        }


# ─────────────────────────────────────────────────────────
# KELAS UTAMA
# ─────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Detektor anomali saham berbasis data historis yfinance.

    Args:
        ticker  : kode saham (misal "BBCA.JK")
        config  : override parameter ANOMALY_CONFIG (opsional)
        fetcher : instance DataFetcher (inject untuk testing)
    """

    def __init__(
        self,
        ticker: str,
        config: Optional[dict] = None,
        fetcher: Optional[DataFetcher] = None,
    ):
        self.ticker = ticker.upper().strip()
        self.cfg = {**ANOMALY_CONFIG, **(config or {})}
        self.fetcher = fetcher or DataFetcher()

    # ─────────────────────────────────────────────
    # PUBLIC — DETEKSI UNTUK SATU TICKER
    # ─────────────────────────────────────────────

    def detect(
        self,
        period_key: str = "1_tahun",
        target_date: Optional[str] = None,
    ) -> list[AnomalyEvent]:
        """
        Jalankan semua deteksi anomali dan kembalikan hasilnya.

        Args:
            period_key  : periode data historis
            target_date : jika diisi (format "YYYY-MM-DD"), hanya return
                          anomali pada tanggal itu. Jika None, semua tanggal.

        Returns:
            List AnomalyEvent, diurutkan dari yang paling baru.
        """
        ohlcv = self.fetcher.get_ohlcv(self.ticker, period_key)

        if len(ohlcv) < self.cfg["min_data_points"]:
            logger.warning(
                f"{self.ticker}: data hanya {len(ohlcv)} baris, "
                f"min {self.cfg['min_data_points']} dibutuhkan."
            )
            return []

        all_events: list[AnomalyEvent] = []

        if self.cfg["detect_price_spike"]:
            all_events.extend(self._detect_price_anomaly(ohlcv))

        if self.cfg["detect_volume_spike"]:
            all_events.extend(self._detect_volume_anomaly(ohlcv))

        if self.cfg["detect_return_anomaly"]:
            all_events.extend(self._detect_return_anomaly(ohlcv))

        if self.cfg["detect_gap"]:
            all_events.extend(self._detect_gap(ohlcv))

        # Filter tanggal jika diminta
        if target_date:
            all_events = [e for e in all_events if str(e.date) == target_date]

        # Deduplicate: jika satu tanggal muncul dari 2 metode, keep yang lebih parah
        all_events = self._deduplicate(all_events)

        # Urutkan: terbaru dulu, lalu severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_events.sort(key=lambda e: (str(e.date), severity_order.get(e.severity, 4)), reverse=True)

        logger.info(f"{self.ticker}: {len(all_events)} anomali ditemukan ({period_key})")
        return all_events

    # ─────────────────────────────────────────────
    # PUBLIC — BATCH UNTUK BANYAK TICKER
    # ─────────────────────────────────────────────

    @staticmethod
    def batch_detect(
        tickers: list[str],
        period_key: str = "1_tahun",
        config: Optional[dict] = None,
        fetcher: Optional[DataFetcher] = None,
    ) -> dict[str, list[AnomalyEvent]]:
        """
        Jalankan deteksi anomali untuk banyak ticker sekaligus.
        Cocok untuk job malam/sore yang dijalankan terjadwal.

        Args:
            tickers    : list ticker
            period_key : periode data
            config     : override config
            fetcher    : DataFetcher instance

        Returns:
            dict mapping ticker → list AnomalyEvent
        """
        fetcher = fetcher or DataFetcher()
        results = {}

        for ticker in tickers:
            try:
                detector = AnomalyDetector(ticker, config=config, fetcher=fetcher)
                results[ticker] = detector.detect(period_key)
            except Exception as e:
                logger.error(f"Gagal deteksi anomali untuk {ticker}: {e}")
                results[ticker] = []

        total = sum(len(v) for v in results.values())
        logger.info(f"Batch selesai: {len(tickers)} ticker, {total} anomali total")
        return results

    @staticmethod
    def print_report(batch_result: dict[str, list[AnomalyEvent]]):
        """Cetak laporan batch deteksi ke stdout."""
        print("=" * 60)
        print("  LAPORAN DETEKSI ANOMALI")
        print("=" * 60)
        total = 0
        for ticker, events in batch_result.items():
            if not events:
                print(f"\n  {ticker}: Tidak ada anomali")
                continue
            print(f"\n  {ticker}: {len(events)} anomali")
            print("  " + "-" * 50)
            for ev in events:
                print("  " + ev.summary().replace("\n", "\n  "))
            total += len(events)
        print(f"\n{'=' * 60}")
        print(f"  Total anomali: {total}")
        print("=" * 60)

    # ─────────────────────────────────────────────
    # PRIVATE — DETEKSI PRICE ANOMALY
    # ─────────────────────────────────────────────

    def _detect_price_anomaly(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        """
        Deteksi lonjakan / penurunan harga ekstrem menggunakan Z-Score rolling.

        Logika:
        - Hitung rolling mean dan rolling std dari harga Close
        - Z-score = (harga_hari_ini - rolling_mean) / rolling_std
        - Jika |z-score| > threshold → anomali
        """
        events = []
        window = self.cfg["rolling_window"]
        threshold = self.cfg["zscore_threshold"]

        close = df["Close"]
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        zscore = (close - rolling_mean) / rolling_std.replace(0, np.nan)

        for dt, z in zscore.items():
            if pd.isna(z) or abs(z) < threshold:
                continue

            mean_val = float(rolling_mean[dt])
            std_val = float(rolling_std[dt])
            actual = float(close[dt])
            lower = mean_val - threshold * std_val
            upper = mean_val + threshold * std_val

            atype = "price_spike" if z > 0 else "price_drop"
            direction = "naik" if z > 0 else "turun"

            events.append(AnomalyEvent(
                ticker=self.ticker,
                date=dt.date(),
                anomaly_type=atype,
                value=actual,
                normal_range=(lower, upper),
                zscore=float(z),
                severity=self._severity_from_zscore(abs(z)),
                description=(
                    f"Harga {direction} signifikan ke {actual:,.2f}. "
                    f"Rentang normal: {lower:,.2f}–{upper:,.2f}. "
                    f"Deviasi: {abs(z):.1f}σ dari rata-rata {window} hari."
                ),
                method="zscore_price",
            ))

        return events

    # ─────────────────────────────────────────────
    # PRIVATE — DETEKSI VOLUME ANOMALY
    # ─────────────────────────────────────────────

    def _detect_volume_anomaly(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        """
        Deteksi lonjakan volume tidak wajar menggunakan IQR dan Z-Score.

        Kombinasi dua metode untuk mengurangi false positive:
        - Z-Score: jika volume jauh dari rata-rata rolling
        - IQR: jika volume di luar fence 1.5×IQR
        Anomali hanya dicatat jika keduanya sepakat.
        """
        events = []
        window = self.cfg["rolling_window"]
        threshold = self.cfg["zscore_threshold"]
        iqr_mult = self.cfg["iqr_multiplier"]

        volume = df["Volume"].astype(float)
        rolling_mean = volume.rolling(window).mean()
        rolling_std = volume.rolling(window).std()
        zscore = (volume - rolling_mean) / rolling_std.replace(0, np.nan)

        # IQR fence global (untuk benchmark)
        q1 = volume.quantile(0.25)
        q3 = volume.quantile(0.75)
        iqr = q3 - q1
        iqr_upper = q3 + iqr_mult * iqr
        iqr_lower = max(0, q1 - iqr_mult * iqr)

        for dt, z in zscore.items():
            if pd.isna(z) or abs(z) < threshold:
                continue

            actual_vol = float(volume[dt])
            # Konfirmasi dengan IQR (volume harus juga di luar IQR fence)
            if actual_vol <= iqr_upper and actual_vol >= iqr_lower:
                continue

            mean_vol = float(rolling_mean[dt])
            std_vol = float(rolling_std[dt])
            lower = max(0, mean_vol - threshold * std_vol)
            upper = mean_vol + threshold * std_vol

            pct_vs_avg = ((actual_vol / mean_vol) - 1) * 100 if mean_vol > 0 else 0

            events.append(AnomalyEvent(
                ticker=self.ticker,
                date=dt.date(),
                anomaly_type="volume_spike",
                value=actual_vol,
                normal_range=(lower, upper),
                zscore=float(z),
                severity=self._severity_from_zscore(abs(z)),
                description=(
                    f"Volume tidak wajar: {actual_vol:,.0f} lot "
                    f"({pct_vs_avg:+.0f}% dari rata-rata {window} hari: {mean_vol:,.0f}). "
                    f"Rentang normal: {lower:,.0f}–{upper:,.0f}."
                ),
                method="zscore_iqr_volume",
                extra={"iqr_upper": iqr_upper, "iqr_lower": iqr_lower},
            ))

        return events

    # ─────────────────────────────────────────────
    # PRIVATE — DETEKSI RETURN ANOMALY
    # ─────────────────────────────────────────────

    def _detect_return_anomaly(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        """
        Deteksi return harian yang sangat ekstrem.

        Return harian = (Close_hari_ini / Close_kemarin) - 1
        Anomali jika return melampaui threshold Z-Score pada distribusi
        rolling return.
        """
        events = []
        window = self.cfg["rolling_window"]
        threshold = self.cfg["zscore_threshold"]

        returns = df["Close"].pct_change().dropna()
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        zscore = (returns - rolling_mean) / rolling_std.replace(0, np.nan)

        for dt, z in zscore.items():
            if pd.isna(z) or abs(z) < threshold:
                continue

            ret_val = float(returns[dt])
            mean_ret = float(rolling_mean[dt])
            std_ret = float(rolling_std[dt])
            lower = mean_ret - threshold * std_ret
            upper = mean_ret + threshold * std_ret

            direction = "positif" if ret_val > 0 else "negatif"

            events.append(AnomalyEvent(
                ticker=self.ticker,
                date=dt.date(),
                anomaly_type="return_anomaly",
                value=ret_val,
                normal_range=(lower, upper),
                zscore=float(z),
                severity=self._severity_from_zscore(abs(z)),
                description=(
                    f"Return harian {direction} ekstrem: {ret_val * 100:+.2f}%. "
                    f"Rentang normal: {lower * 100:.2f}% – {upper * 100:.2f}%. "
                    f"Deviasi: {abs(z):.1f}σ."
                ),
                method="zscore_return",
            ))

        return events

    # ─────────────────────────────────────────────
    # PRIVATE — DETEKSI GAP
    # ─────────────────────────────────────────────

    def _detect_gap(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        """
        Deteksi gap harga: selisih Open hari ini vs Close kemarin.

        Gap terjadi ketika harga pembukaan berbeda jauh dari penutupan
        hari sebelumnya, biasanya karena berita besar setelah jam bursa.

        Gap up  : Open > Close_kemarin + threshold%
        Gap down: Open < Close_kemarin - threshold%
        """
        events = []
        threshold_pct = self.cfg["gap_threshold_pct"] / 100

        prev_close = df["Close"].shift(1)
        gap_pct = (df["Open"] - prev_close) / prev_close.replace(0, np.nan)

        for dt, gap in gap_pct.items():
            if pd.isna(gap) or abs(gap) < threshold_pct:
                continue

            open_price = float(df.loc[dt, "Open"])
            prev_c = float(prev_close[dt])
            atype = "gap_up" if gap > 0 else "gap_down"
            direction = "naik" if gap > 0 else "turun"

            # Z-score untuk gap (relatif terhadap distribusi gap historis)
            gap_std = float(gap_pct.std())
            gap_mean = float(gap_pct.mean())
            z = (gap - gap_mean) / gap_std if gap_std > 1e-9 else 0

            events.append(AnomalyEvent(
                ticker=self.ticker,
                date=dt.date(),
                anomaly_type=atype,
                value=gap,
                normal_range=(-threshold_pct, threshold_pct),
                zscore=float(z),
                severity=self._severity_from_gap(abs(gap)),
                description=(
                    f"Gap {direction} {abs(gap) * 100:.2f}%: "
                    f"Open {open_price:,.2f} vs Close kemarin {prev_c:,.2f}. "
                    f"Biasanya dipicu berita besar / corporate action."
                ),
                method="gap_detector",
                extra={"open_price": open_price, "prev_close": prev_c},
            ))

        return events

    # ─────────────────────────────────────────────
    # PRIVATE — HELPERS
    # ─────────────────────────────────────────────

    def _severity_from_zscore(self, abs_z: float) -> str:
        """Tentukan severity berdasarkan nilai absolut z-score."""
        if abs_z >= 4.0:
            return "critical"
        elif abs_z >= 3.0:
            return "high"
        elif abs_z >= 2.5:
            return "medium"
        else:
            return "low"

    def _severity_from_gap(self, abs_gap_pct: float) -> str:
        """Tentukan severity berdasarkan besar gap."""
        if abs_gap_pct >= 0.10:    # >= 10%
            return "critical"
        elif abs_gap_pct >= 0.07:  # >= 7%
            return "high"
        elif abs_gap_pct >= 0.05:  # >= 5%
            return "medium"
        else:
            return "low"

    def _deduplicate(self, events: list[AnomalyEvent]) -> list[AnomalyEvent]:
        """
        Hapus duplikat: jika tanggal + anomaly_type sama dari metode berbeda,
        simpan hanya yang severity-nya paling tinggi.
        """
        seen: dict[tuple, AnomalyEvent] = {}
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        for ev in events:
            key = (ev.date, ev.anomaly_type)
            if key not in seen:
                seen[key] = ev
            else:
                # Simpan yang lebih parah
                existing = seen[key]
                if severity_order.get(ev.severity, 4) < severity_order.get(existing.severity, 4):
                    seen[key] = ev

        return list(seen.values())

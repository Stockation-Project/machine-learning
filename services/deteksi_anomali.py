
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

from core.data_fetcher import DataFetcher
from config.settings import ANOMALY_CONFIG

# Import ReGEN-TAD internal detector
from regen_tad_detector import ReGENTADDetector, InternalAnomalyEvent

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC AnomalyEvent — SAMA PERSIS DENGAN VERSI LAMA
# =============================================================================
@dataclass
class AnomalyEvent:
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

    def summary(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.ticker} | {self.date} | ReGEN-TAD\n"
            f" Score  : {self.value:.4f}\n"
            f" Z-Score: {self.zscore:.2f}σ\n"
            f" Ket.   : {self.description}\n"
        )

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": str(self.date),
            "anomaly_type": self.anomaly_type,
            "value": round(self.value, 6),
            "normal_range": {
                "lower": round(self.normal_range[0], 6),
                "upper": round(self.normal_range[1], 6)
            },
            "zscore": round(self.zscore, 4),
            "severity": self.severity,
            "description": self.description,
            "method": self.method,
            "extra": self.extra,
        }


# =============================================================================
# PUBLIC DETECTOR — INTERFACE SAMA DENGAN VERSI LAMA
# =============================================================================
class AnomalyDetector:
    """Kelas utama yang dipakai di seluruh sistem"""

    def __init__(
        self,
        ticker: str,
        config: Optional[dict] = None,
        fetcher: Optional[DataFetcher] = None,
    ):
        self.ticker = ticker.upper().strip()
        self.cfg = {**ANOMALY_CONFIG, **(config or {})}
        self.fetcher = fetcher or DataFetcher()
        
        self._model_path = f"./models/{self.ticker}_regentad.pt"
        self._internal_detector = None

    def detect(
        self, period_key: str = "1_tahun", target_date: Optional[str] = None
    ) -> List[AnomalyEvent]:
        
        df = self.fetcher.get_ohlcv(self.ticker, period_key)
        
        if len(df) < self.cfg.get("min_data_points", 100):
            logger.warning(f"{self.ticker}: Data terlalu sedikit ({len(df)} baris)")
            return []

        # Inisialisasi / Load ReGEN-TAD Detector
        if self._internal_detector is None:
            self._internal_detector = self._init_regentad_detector(df)

        # Deteksi menggunakan ReGEN-TAD
        internal_result = self._internal_detector.detect(df)

        # Konversi ke AnomalyEvent public
        events: List[AnomalyEvent] = []
        for iev in internal_result.events:
            event = AnomalyEvent(
                ticker=iev.ticker,
                date=pd.to_datetime(iev.date).date(),
                anomaly_type="regen_tad",
                value=iev.anomaly_score,
                normal_range=(0.0, iev.anomaly_score * 0.8),
                zscore=iev.anomaly_score,
                severity=iev.severity,
                description=iev.description,
                method="ReGEN-TAD",
                extra={
                    "attribution": iev.attribution,
                    "diagnostics": iev.diagnostics,
                }
            )
            events.append(event)

        if target_date:
            events = [e for e in events if str(e.date) == target_date]

        events.sort(key=lambda e: e.zscore, reverse=True)
        return events

    def _init_regentad_detector(self, df: pd.DataFrame):
        """Inisialisasi atau load model ReGEN-TAD"""
        Path("./models").mkdir(parents=True, exist_ok=True)

        detector = ReGENTADDetector(self.ticker, config=self.cfg.get("regentad_config", {}))

        if Path(self._model_path).exists():
            detector.load(self._model_path)
        else:
            logger.info(f"🔄 Training ReGEN-TAD model untuk {self.ticker}... (bisa memakan waktu)")
            detector.fit(df, save_path=self._model_path)

        return detector

    # Batch & Report tetap sama
    @staticmethod
    def batch_detect(
        tickers: List[str],
        period_key: str = "1_tahun",
        config: Optional[dict] = None,
        fetcher: Optional[DataFetcher] = None,
    ) -> Dict[str, List[AnomalyEvent]]:
        fetcher = fetcher or DataFetcher()
        results = {}
        for ticker in tickers:
            try:
                detector = AnomalyDetector(ticker, config=config, fetcher=fetcher)
                results[ticker] = detector.detect(period_key)
            except Exception as e:
                logger.error(f"Gagal deteksi {ticker}: {e}")
                results[ticker] = []
        return results

    @staticmethod
    def print_report(batch_result: Dict[str, List[AnomalyEvent]]):
        print("=" * 70)
        print("  LAPORAN DETEKSI ANOMALI — ReGEN-TAD")
        print("=" * 70)
        total = 0
        for ticker, events in batch_result.items():
            if not events:
                print(f"\n  {ticker}: Tidak ada anomali")
                continue
            print(f"\n  {ticker}: {len(events)} anomali")
            print("  " + "-" * 60)
            for ev in events:
                print("  " + ev.summary().replace("\n", "\n  "))
            total += len(events)
        print(f"\n{'=' * 70}")
        print(f"  Total anomali: {total}")
        print("=" * 70)


# Untuk backward compatibility
__all__ = ["AnomalyDetector", "AnomalyEvent"]
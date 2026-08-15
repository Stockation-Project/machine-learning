# core/data_fetcher.py
import yfinance as yf
import pandas as pd


# Bandingkan slider risiko dengan snapshot harga yang sama. Tanpa cache ini,
# setiap geseran mengunduh ulang data dan metrik CVaR dapat berubah meski bobot
# alokasi tidak berubah.
_price_cache: dict[tuple[tuple[str, ...], str], pd.DataFrame] = {}

class DataFetcher:
    def _get_yfinance_period(self, period_key: str) -> str:
        # Mengubah format key ke format yang dikenali yfinance
        mapping = {
            "1_bulan": "1mo",
            "3_bulan": "3mo",
            "6_bulan": "6mo",
            "1_tahun": "1y",
            "3_tahun": "3y",
            "5_tahun": "5y",
            "10_tahun": "10y",
        }
        return mapping.get(period_key, "1y")

    def get_ohlcv(self, ticker: str, period_key: str) -> pd.DataFrame:
        """Mengambil data OHLCV untuk satu saham."""
        period = self._get_yfinance_period(period_key)
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df

    def get_prices(self, tickers: list[str], period_key: str) -> pd.DataFrame:
        """Mengambil harga Close untuk banyak saham sekaligus untuk optimasi."""
        if not tickers:
            return pd.DataFrame()

        cache_key = (tuple(tickers), period_key)
        cached = _price_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
            
        period = self._get_yfinance_period(period_key)
        # Unduh data secara massal tanpa memunculkan progress bar
        data = yf.download(tickers, period=period, progress=False)
        
        # yfinance mengembalikan format yang berbeda jika hanya 1 saham vs banyak saham
        if len(tickers) == 1:
            df = data[['Close']].copy()
            df.columns = [tickers[0]]
            _price_cache[cache_key] = df.copy()
            return df
        else:
            df = data['Close'].copy()
            _price_cache[cache_key] = df.copy()
            return df

import pandas as pd
import yfinance as yf

from src.config import (
    IDX80_TICKERS,
    RAW_DATA_PATH,
    START_DATE,
)

from src.logger import setup_logger

logger = setup_logger()


class IDX80DataUpdater:

    def __init__(self):
        self.tickers = IDX80_TICKERS

    def download_data(self, start_date=START_DATE):

        logger.info("Downloading stock data...")

        all_data = []

        for ticker in self.tickers:

            logger.info(f"Downloading {ticker}")

            try:

                df = yf.download(
                    ticker,
                    start=start_date,
                    auto_adjust=True,
                    progress=False,
                )

                # FIX MULTIINDEX COLUMNS
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if df.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue

                # Keep only close
                df = df[["Close"]].copy()

                # Reset index
                df = df.reset_index()

                # Add ticker
                df["Ticker"] = ticker

                all_data.append(df)

            except Exception as e:
                logger.error(f"Failed downloading {ticker}: {e}")

        combined = pd.concat(all_data, ignore_index=True)

        combined["Date"] = pd.to_datetime(combined["Date"])

        logger.info("Download complete")

        return combined
    
    def update_master_dataset(self):

        # Load existing historical data
        historical_data = pd.read_parquet(
            RAW_DATA_PATH
        )
        
        historical_data["Date"] = pd.to_datetime(
            historical_data["Date"]
        )

        # Find latest available date
        latest_date = historical_data["Date"].max()

        logger.info(
            f"Latest historical date: {latest_date}"
        )

        # Download only newer data
        new_data = self.download_data(
            start_date=(latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )

        # Combine datasets
        combined = pd.concat(
            [historical_data, new_data],
            ignore_index=True
        )

        # Remove duplicates
        combined = combined.drop_duplicates(
            subset=["Date", "Ticker"]
        )

        # Sort properly
        combined = combined.sort_values(
            ["Ticker", "Date"]
        )

        # Save updated dataset
        combined.to_parquet(
            RAW_DATA_PATH,
            index=False
        )

        logger.info("Dataset updated successfully")

        return combined
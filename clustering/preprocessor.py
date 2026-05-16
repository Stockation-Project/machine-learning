# src/preprocessing.py

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler


class SequencePreprocessor:

    def __init__(
        self,
        seq_length=60,
        scaler_path="models/sequence_scaler.pkl",
    ):

        self.seq_length = seq_length
        self.scaler_path = scaler_path
        self.scaler = StandardScaler()

    def create_sequences(self, df):

        """
        Create rolling log-return sequences for each stock.
        """

        sequences = []
        tickers = []
        end_dates = []

        for ticker, group in df.groupby("Ticker"):

            group = group.copy()

            # Sort chronologically
            group = group.sort_values("Date")

            # Calculate log returns
            group["log_return"] = np.log(
                group["Close"] / group["Close"].shift(1)
            )

            # Remove NaN from first return
            group = group.dropna(subset=["log_return"])

            returns = group["log_return"].values
            dates = group["Date"].values

            # Rolling window sequences
            for i in range(len(returns) - self.seq_length + 1):

                seq = returns[i:i + self.seq_length]

                sequences.append(seq)

                tickers.append(ticker)

                end_dates.append(
                    dates[i + self.seq_length - 1]
                )

        X_seq = np.array(sequences)

        return (
            X_seq,
            np.array(tickers, dtype=object),
            np.array(end_dates),
        )

    def fit_transform(self, X_seq):

        """
        Fit scaler and normalize sequences.
        """

        original_shape = X_seq.shape

        X_scaled = self.scaler.fit_transform(
            X_seq.reshape(-1, 1)
        ).reshape(original_shape)

        return X_scaled

    def transform(self, X_seq):

        """
        Transform using existing scaler.
        """

        original_shape = X_seq.shape

        X_scaled = self.scaler.transform(
            X_seq.reshape(-1, 1)
        ).reshape(original_shape)

        return X_scaled

    def save_scaler(self):

        """
        Save fitted scaler.
        """

        joblib.dump(
            self.scaler,
            self.scaler_path,
        )

    def load_scaler(self):

        """
        Load existing scaler.
        """

        self.scaler = joblib.load(
            self.scaler_path
        )

    def validate_sequences(self, X_seq):

        """
        Basic validation checks.
        """

        print("Sequence Shape:", X_seq.shape)

        print("NaN Count:", np.isnan(X_seq).sum())

        print("Inf Count:", np.isinf(X_seq).sum())

        print("Mean:", X_seq.mean())

        print("Std:", X_seq.std())
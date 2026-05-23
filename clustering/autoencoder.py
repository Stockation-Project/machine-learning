# src/autoencoder.py

import pandas as pd
import tensorflow as tf


class CNNAutoencoder:

    def __init__(self):

        self.autoencoder = None
        self.encoder = None

    def load_models(
        self,
        autoencoder_path,
        encoder_path,
    ):

        """
        Load trained autoencoder and encoder.
        """

        self.autoencoder = (
            tf.keras.models.load_model(
                autoencoder_path
            )
        )

        self.encoder = (
            tf.keras.models.load_model(
                encoder_path
            )
        )

    def extract_embeddings(self, X):

        """
        Generate latent embeddings
        using trained encoder.
        """

        embeddings = self.encoder.predict(X)

        return embeddings

    def aggregate_stock_embeddings(
        self,
        embeddings,
        tickers,
    ):

        """
        Aggregate sequence embeddings
        into stock-level embeddings.
        """

        latent_df = pd.DataFrame(
            embeddings
        )

        latent_df["Ticker"] = tickers

        stock_embeddings = (
            latent_df
            .groupby("Ticker")
            .mean()
        )

        return stock_embeddings
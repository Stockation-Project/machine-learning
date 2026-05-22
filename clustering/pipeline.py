import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import components from clustering folder
from clustering.preprocessor import SequencePreprocessor
from clustering.autoencoder import CNNAutoencoder
from clustering.clustering import StockClusterModel


class ClusteringService:

    def __init__(self):

        # Base path
        self.base_path = (
            Path(__file__).resolve().parent.parent / "clustering"
        )

        # Paths
        self.paths = {
            "autoencoder": self.base_path / "autoencoder.keras",
            "encoder": self.base_path / "encoder.keras",
            "kmeans": self.base_path / "kmeans.pkl",
            "latent_scaler": self.base_path / "latent_scaler.pkl",
            "sequence_scaler": self.base_path / "sequence_scaler.pkl",
        }

    def classify_all_stocks(self):

        """
        Update dataset, preprocess,
        extract embeddings, predict clusters,
        and map clusters to risk levels.

        Returns:
            {
                "BBCA.JK": "Low",
                "BBRI.JK": "Medium",
                ...
            }
        """

        try:

            # ======================================
            # UPDATE DATASET
            # ======================================

            updater = IDX80DataUpdater()

            df = updater.update_master_dataset()

            # Ensure datetime
            df["Date"] = pd.to_datetime(
                df["Date"]
            )

            # ======================================
            # PREPROCESS
            # ======================================

            preprocessor = SequencePreprocessor(
                seq_length=60,
                scaler_path=str(
                    self.paths["sequence_scaler"]
                )
            )

            # IMPORTANT:
            # Load EXISTING scaler
            preprocessor.load_scaler()

            X_seq, tickers, end_dates = (
                preprocessor.create_sequences(df)
            )

            # Transform only
            X_scaled = preprocessor.transform(
                X_seq
            )

            # IMPORTANT:
            # Add channel dimension for CNN
            X_scaled = X_scaled[..., None]

            # ======================================
            # LOAD ENCODER
            # ======================================

            model = CNNAutoencoder()

            model.load_models(
                autoencoder_path=str(
                    self.paths["autoencoder"]
                ),
                encoder_path=str(
                    self.paths["encoder"]
                ),
            )

            # ======================================
            # EXTRACT EMBEDDINGS
            # ======================================

            embeddings = model.extract_embeddings(
                X_scaled
            )

            # ======================================
            # AGGREGATE STOCK EMBEDDINGS
            # ======================================

            stock_embeddings = (
                model.aggregate_stock_embeddings(
                    embeddings,
                    tickers
                )
            )

            # ======================================
            # LOAD CLUSTER MODEL
            # ======================================

            cluster_model = StockClusterModel()

            cluster_model.load_models(
                kmeans_path=str(
                    self.paths["kmeans"]
                ),
                scaler_path=str(
                    self.paths["latent_scaler"]
                )
            )

            # ======================================
            # PREDICT CLUSTERS
            # ======================================

            labels = cluster_model.predict(
                stock_embeddings.values
            )

            # ======================================
            # CREATE RESULT DATAFRAME
            # ======================================

            result_df = pd.DataFrame({
                "ticker": stock_embeddings.index,
                "cluster": labels
            })

            # ======================================
            # CALCULATE VOLATILITY
            # ======================================

            stock_volatility = (
                df.groupby("Ticker")["Close"]
                .pct_change()
                .groupby(df["Ticker"])
                .std()
            )

            # ======================================
            # JOIN VOLATILITY
            # ======================================

            result_with_vol = (
                result_df
                .set_index("ticker")
                .join(
                    stock_volatility.rename(
                        "volatility"
                    )
                )
            )

            # ======================================
            # MEAN VOLATILITY PER CLUSTER
            # ======================================

            cluster_vol = (
                result_with_vol
                .groupby("cluster")["volatility"]
                .mean()
                .sort_values()
            )

            # ======================================
            # MAP CLUSTERS TO RISK LEVEL
            # ======================================

            risk_mapping = {
                cluster_vol.index[0]: "Low",
                cluster_vol.index[1]: "Medium",
                cluster_vol.index[2]: "High"
            }

            result_df["risk_level"] = (
                result_df["cluster"]
                .map(risk_mapping)
            )

            # ======================================
            # RETURN FINAL DICTIONARY
            # ======================================

            return (
                result_df
                .set_index("ticker")["risk_level"]
                .to_dict()
            )

        except Exception as e:

            print(
                f"Error in clustering service: {str(e)}"
            )

            raise e
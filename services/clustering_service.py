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
        # Base path to clustering directory
        self.base_path = Path(__file__).resolve().parent.parent / "clustering"
        
        # Paths to models and data
        self.paths = {
            "autoencoder": self.base_path / "autoencoder.keras",
            "encoder": self.base_path / "encoder.keras",
            "kmeans": self.base_path / "kmeans.pkl",
            "latent_scaler": self.base_path / "latent_scaler.pkl",
            "sequence_scaler": self.base_path / "sequence_scaler.pkl",
            "raw_data": self.base_path / "idx80_raw.parquet"
        }

    def classify_all_stocks(self):
        """
        Processes historical data and returns a mapping of ticker -> risk_level
        """
        try:
            # 1. Load Data
            if not self.paths["raw_data"].exists():
                raise FileNotFoundError(f"Data file not found at {self.paths['raw_data']}")
            
            try:
                # Coba load dengan engine fastparquet karena pyarrow sering error 'No type extension'
                df = pd.read_parquet(str(self.paths["raw_data"]), engine='fastparquet')
            except Exception as e:
                print(f"Fastparquet failed, trying pyarrow: {e}")
                df = pd.read_parquet(str(self.paths["raw_data"]))

            
            # 2. Preprocess
            # We use the same settings as trained (seq_length=60 based on model error)
            preprocessor = SequencePreprocessor(seq_length=60, scaler_path=str(self.paths["sequence_scaler"]))
            preprocessor.load_scaler()
            
            X_seq, tickers, _ = preprocessor.create_sequences(df)
            X_scaled = preprocessor.transform(X_seq)
            
            # 3. Extract Embeddings
            ae = CNNAutoencoder()
            ae.load_models(
                autoencoder_path=str(self.paths["autoencoder"]),
                encoder_path=str(self.paths["encoder"])
            )
            
            embeddings = ae.extract_embeddings(X_scaled)
            stock_embeddings = ae.aggregate_stock_embeddings(embeddings, tickers)
            
            # 4. Cluster
            cluster_model = StockClusterModel()
            cluster_model.load_models(
                kmeans_path=str(self.paths["kmeans"]),
                scaler_path=str(self.paths["latent_scaler"])
            )
            
            labels = cluster_model.predict(stock_embeddings.values)
            
            # 5. Map Cluster to Risk Level based on Volatility
            # Calculate volatility for each stock to help rank clusters
            stock_volatility = df.groupby("Ticker")["Close"].pct_change().groupby(df["Ticker"]).std()
            
            result_df = pd.DataFrame({
                "ticker": stock_embeddings.index,
                "cluster": labels
            })
            
            # Join with volatility to calculate mean volatility per cluster
            result_with_vol = result_df.set_index("ticker").join(stock_volatility.rename("volatility"))
            cluster_vol = result_with_vol.groupby("cluster")["volatility"].mean().sort_values()
            
            # Map sorted clusters to labels
            # Smallest volatility = Low, Middle = Medium, Highest = High
            risk_mapping = {
                cluster_vol.index[0]: "Low",
                cluster_vol.index[1]: "Medium",
                cluster_vol.index[2]: "High"
            }
            
            result_df["risk_level"] = result_df["cluster"].map(risk_mapping)
            
            # Convert to dictionary { "BBCA": "Low", ... }
            return result_df.set_index("ticker")["risk_level"].to_dict()

        except Exception as e:
            print(f"Error in clustering service: {str(e)}")
            raise e

import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class StockClusterModel:

    def __init__(
        self,
        n_clusters=3,
        random_state=42,
    ):

        self.n_clusters = n_clusters

        self.scaler = StandardScaler()

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            init="k-means++",
            n_init="auto",
        )

    def fit_predict(self, stock_embeddings):

        """
        Scale embeddings and fit KMeans.
        """

        latent_scaled = (
            self.scaler.fit_transform(
                stock_embeddings
            )
        )

        labels = self.model.fit_predict(
            latent_scaled
        )

        return labels

    def predict(self, stock_embeddings):

        """
        Predict cluster labels only.
        """

        latent_scaled = (
            self.scaler.transform(
                stock_embeddings
            )
        )

        labels = self.model.predict(
            latent_scaled
        )

        return labels

    def save_models(
        self,
        kmeans_path="models/kmeans.pkl",
        scaler_path="models/latent_scaler.pkl",
    ):

        """
        Save clustering model and scaler.
        """

        joblib.dump(
            self.model,
            kmeans_path
        )

        joblib.dump(
            self.scaler,
            scaler_path
        )

    def load_models(
        self,
        kmeans_path="models/kmeans.pkl",
        scaler_path="models/latent_scaler.pkl",
    ):

        """
        Load clustering model and scaler.
        """

        self.model = joblib.load(
            kmeans_path
        )

        self.scaler = joblib.load(
            scaler_path
        )
# ============================================================
# LOADER
# Load model + artifacts once at API startup
# ============================================================

import json
import joblib
import tensorflow as tf

from src.config import (
    MODEL_PATH,
    STOCK_SCALERS_PATH,
    TICKER_ENCODER_PATH,
    SECTOR_ENCODER_PATH,
    INDUSTRY_ENCODER_PATH,
    FEATURE_INFO_PATH,
)


# ============================================================
# CUSTOM LAYER
# Must be defined here so load_model() can find it
# ============================================================

class TemporalAttentionPooling(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = tf.keras.layers.Dense(1)
        super().build(input_shape)

    def call(self, inputs):
        scores  = self.attention_dense(inputs)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts() -> dict:
    """
    Load all deployment artifacts.
    Call once at startup and pass the result around.
    """

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TemporalAttentionPooling": TemporalAttentionPooling},
    )

    stock_scalers    = joblib.load(STOCK_SCALERS_PATH)
    ticker_encoder   = joblib.load(TICKER_ENCODER_PATH)
    sector_encoder   = joblib.load(SECTOR_ENCODER_PATH)
    industry_encoder = joblib.load(INDUSTRY_ENCODER_PATH)

    with open(FEATURE_INFO_PATH) as f:
        feature_info = json.load(f)

    return {
        "model":            model,
        "stock_scalers":    stock_scalers,
        "ticker_encoder":   ticker_encoder,
        "sector_encoder":   sector_encoder,
        "industry_encoder": industry_encoder,
        "feature_info":     feature_info,
    }

# ============================================================
# SCHEMAS
# Pydantic request / response models
# ============================================================

from pydantic import BaseModel, Field


# ============================================================
# REQUEST
# ============================================================

class PredictRequest(BaseModel):

    ticker: str = Field(
        ...,
        description="IDX ticker with .JK suffix, e.g. 'BBCA.JK'",
        examples=["BBCA.JK"],
    )

    sector: str = Field(
        ...,
        description="Sector string exactly as in training data, e.g. 'Financials'",
        examples=["Financials"],
    )

    industry: str = Field(
        ...,
        description="Industry string exactly as in training data, e.g. 'Banks'",
        examples=["Banks"],
    )


class BatchPredictRequest(BaseModel):

    stocks: list[PredictRequest] = Field(
        ...,
        description="List of stocks to predict",
    )


# ============================================================
# RESPONSE
# ============================================================

class HorizonResult(BaseModel):
    predicted_volatility: float
    price_low:            float
    price_high:           float


class PredictResponse(BaseModel):
    ticker:        str
    current_price: float
    forecast_10d:  HorizonResult
    forecast_1m:   HorizonResult
    forecast_3m:   HorizonResult


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    errors:  dict[str, str] = {}


# ============================================================
# HELPER — convert predictor output dict to response schema
# ============================================================

def to_response(pred: dict) -> PredictResponse:

    return PredictResponse(
        ticker=pred["ticker"],
        current_price=pred["current_price"],
        forecast_10d=HorizonResult(
            predicted_volatility=pred["pred_vol_10d"],
            price_low=pred["range_10d_low"],
            price_high=pred["range_10d_high"],
        ),
        forecast_1m=HorizonResult(
            predicted_volatility=pred["pred_vol_1m"],
            price_low=pred["range_1m_low"],
            price_high=pred["range_1m_high"],
        ),
        forecast_3m=HorizonResult(
            predicted_volatility=pred["pred_vol_3m"],
            price_low=pred["range_3m_low"],
            price_high=pred["range_3m_high"],
        ),
    )

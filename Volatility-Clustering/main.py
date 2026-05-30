# ============================================================
# MAIN — FastAPI app
# ============================================================

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.loader    import load_artifacts
from src.predictor import predict_one
from src.schemas   import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    to_response,
)


# ============================================================
# LIFESPAN — load model once at startup
# ============================================================

artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model artifacts...")
    artifacts.update(load_artifacts())
    print("Model ready.")
    yield
    artifacts.clear()


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="IDX Volatility Forecaster",
    description=(
        "Multi-horizon volatility prediction (10D / 1M / 3M) "
        "for IDX80 stocks, with estimated price range per horizon."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(artifacts)}


# ============================================================
# SINGLE PREDICTION
# POST /predict
# ============================================================

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict volatility and price range for one stock.

    Example request body:
    ```json
    {
        "ticker":   "BBCA.JK",
        "sector":   "Financials",
        "industry": "Banks"
    }
    ```

    Example response:
    ```json
    {
        "ticker": "BBCA.JK",
        "current_price": 9500.0,
        "forecast_10d": {
            "predicted_volatility": 0.012,
            "price_low": 9385.0,
            "price_high": 9617.0
        },
        "forecast_1m": { ... },
        "forecast_3m": { ... }
    }
    ```
    """

    try:
        result = predict_one(
            ticker=request.ticker,
            sector=request.sector,
            industry=request.industry,
            artifacts=artifacts,
        )
        return to_response(result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================
# BATCH PREDICTION
# POST /predict/batch
# ============================================================

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    """
    Predict for multiple stocks in one call.
    Stocks that fail are collected in `errors` and do not
    cause the whole batch to fail.
    """

    results = []
    errors  = {}

    for stock in request.stocks:
        try:
            result = predict_one(
                ticker=stock.ticker,
                sector=stock.sector,
                industry=stock.industry,
                artifacts=artifacts,
            )
            results.append(to_response(result))

        except Exception as e:
            errors[stock.ticker] = str(e)

    return BatchPredictResponse(results=results, errors=errors)


# ============================================================
# LIST VALID TICKERS / SECTORS / INDUSTRIES
# GET /metadata
# ============================================================

@app.get("/metadata")
def metadata():
    """
    Return all valid ticker, sector, and industry values
    as stored in the label encoders.
    """

    if not artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    return {
        "tickers":    list(artifacts["ticker_encoder"].classes_),
        "sectors":    list(artifacts["sector_encoder"].classes_),
        "industries": list(artifacts["industry_encoder"].classes_),
    }

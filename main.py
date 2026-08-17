# Konfigurasi SSL CA bundle PALING AWAL — sebelum import apa pun yang
# melakukan koneksi HTTPS (yfinance, requests, supabase). Ini mengatasi
# error verifikasi sertifikat di lingkungan dengan antivirus/proxy TLS.
import core.ssl_setup  # noqa: F401  (side-effect import)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import contextlib

# Mengimpor class yang sudah dibuat tim AI-mu
from services.deteksi_anomali import AnomalyDetector
from services.optimasi_portofolio import PortfolioOptimizer
from services.clustering_service import ClusteringService
from services.ai_explanation import AIExplanationService

from services.volatility.loader import load_artifacts
from services.volatility.predictor import predict_one
from services.volatility.schemas import PredictRequest, PredictResponse, to_response

volatility_artifacts = {}

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Volatility model artifacts...")
    volatility_artifacts.update(load_artifacts())
    print("Volatility Model ready.")
    yield
    volatility_artifacts.clear()

app = FastAPI(
    title="Stockation AI API",
    description="Microservice AI untuk memproses anomali dan optimasi portofolio",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------
# DTO (Data Transfer Object) - Padanan req.body di Express
# ---------------------------------------------------------
class OptimizeRequest(BaseModel):
    tickers: List[str]
    risk_profile: str = "moderat"
    risk_tolerance: Optional[float] = None
    period_key: str = "1_tahun"
    method: Optional[str] = None

class RecommendRequest(BaseModel):
    universe: List[str]
    top_n: int = 5
    risk_profile: str = "moderat"
    risk_tolerance: Optional[float] = None
    period_key: str = "1_tahun"

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------

@app.get("/api/ai/anomalies/{ticker}")
def get_anomalies(ticker: str, period_key: str = "1_tahun"):
    """
    Endpoint untuk mendapatkan histori anomali dari satu saham (Dipakai di Halaman Detail Saham)
    """
    try:
        detector = AnomalyDetector(ticker=ticker)
        events = detector.detect(period_key=period_key)
        
        return {
            "success": True,
            "ticker": ticker,
            # Class dari timmu sudah punya method .to_dict(), ini sangat memudahkan!
            "anomalies": [ev.to_dict() for ev in events] 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mendeteksi anomali: {str(e)}")

@app.post("/api/ai/optimize")
def optimize_portfolio(req: OptimizeRequest):
    """
    Endpoint untuk menghitung bobot optimal dari saham yang dipilih user.
    Risiko memakai volatilitas PREDIKSI (model LSTM) bila artefak tersedia.
    """
    try:
        optimizer = PortfolioOptimizer(
            risk_profile=req.risk_profile,
            risk_tolerance=req.risk_tolerance,
            vol_artifacts=volatility_artifacts or None,
        )
        result = optimizer.optimize(
            tickers=req.tickers,
            period_key=req.period_key,
            method="mean_cvar",
        )
        
        return {
            "success": True,
            "data": result.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/ai/clustering")
def get_stock_clustering():
    """
    Endpoint untuk mendapatkan klasifikasi resiko saham (Low, Medium, High)
    berdasarkan model clustering yang sudah dilatih.
    """
    try:
        service = ClusteringService()
        mapping = service.classify_all_stocks()
        return {
            "success": True,
            "data": mapping
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/recommend")
def recommend_portfolio(req: RecommendRequest):
    """
    Endpoint untuk meminta sistem memilihkan saham terbaik + bobotnya (Dipakai saat create portofolio)
    """
    try:
        optimizer = PortfolioOptimizer(
            risk_profile=req.risk_profile,
            risk_tolerance=req.risk_tolerance
        )
        result = optimizer.recommend(
            universe=req.universe, 
            top_n=req.top_n, 
            period_key=req.period_key
        )
        
        return {
            "success": True,
            "data": result.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class ExplainRequest(BaseModel):
    term: str
    context: Optional[str] = None

@app.post("/api/ai/explain")
def explain_term(req: ExplainRequest):
    """
    Endpoint untuk mendapatkan penjelasan dari AI (Gemini) terkait istilah saham.
    """
    try:
        service = AIExplanationService()
        explanation = service.explain_term(req.term, req.context)
        return {
            "success": True,
            "data": {
                "term": req.term,
                "context": req.context,
                "explanation": explanation
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/volatility", response_model=PredictResponse)
def get_volatility_prediction(req: PredictRequest):
    """
    Endpoint untuk memprediksi risiko / volatilitas saham dalam 10D, 1M, 3M
    """
    try:
        if not volatility_artifacts:
            raise HTTPException(status_code=503, detail="Model belum siap.")
            
        result = predict_one(
            ticker=req.ticker,
            sector=req.sector,
            industry=req.industry,
            artifacts=volatility_artifacts,
        )
        return to_response(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


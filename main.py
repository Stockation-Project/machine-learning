from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Mengimpor class yang sudah dibuat tim AI-mu
from services.deteksi_anomali import AnomalyDetector
from services.optimasi_portofolio import PortfolioOptimizer

app = FastAPI(
    title="Stockation AI API",
    description="Microservice AI untuk memproses anomali dan optimasi portofolio",
    version="1.0.0"
)

# ---------------------------------------------------------
# DTO (Data Transfer Object) - Padanan req.body di Express
# ---------------------------------------------------------
class OptimizeRequest(BaseModel):
    tickers: List[str]
    risk_profile: str = "moderat"
    risk_tolerance: Optional[float] = None
    period_key: str = "1_tahun"

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
    Endpoint untuk menghitung bobot optimal dari saham yang dipilih user
    """
    try:
        optimizer = PortfolioOptimizer(
            risk_profile=req.risk_profile,
            risk_tolerance=req.risk_tolerance
        )
        result = optimizer.optimize(tickers=req.tickers, period_key=req.period_key)
        
        return {
            "success": True,
            "data": result.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
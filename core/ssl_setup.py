"""
ssl_setup.py
============
Memastikan koneksi HTTPS keluar (yfinance/curl_cffi, requests, supabase, dll)
dapat memverifikasi sertifikat di lingkungan yang memakai antivirus atau proxy
penyadap TLS (mis. Kaspersky, ESET, Bitdefender, proxy korporat/kampus).

Masalah yang diatasi:
    "unable to get local issuer certificate" / "UNABLE_TO_VERIFY_LEAF_SIGNATURE"

Cara kerja:
    1. Bangun CA bundle gabungan = certifi (CA publik) + root CA dari
       Windows certificate store (termasuk CA yang disuntik antivirus/proxy).
    2. Set environment variable agar dipakai oleh library jaringan:
         - CURL_CA_BUNDLE -> libcurl (curl_cffi / yfinance)
         - SSL_CERT_FILE  -> ssl / requests bawaan Python
         - REQUESTS_CA_BUNDLE -> requests

PENTING: impor modul ini PALING AWAL (sebelum yfinance/requests dipakai).
"""
import os
import ssl
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent
_BUNDLE_PATH = _BASE_DIR / "artifacts" / "system_ca_bundle.pem"


def _build_bundle() -> Path | None:
    """Bangun CA bundle gabungan certifi + Windows store. Return path bundle."""
    try:
        import certifi
    except ImportError:
        logger.warning("certifi tidak tersedia; lewati pembuatan CA bundle.")
        return None

    parts = [Path(certifi.where()).read_text(encoding="utf-8").strip()]

    # Ambil sertifikat dari Windows cert store (hanya tersedia di Windows)
    if hasattr(ssl, "enum_certificates"):
        seen: set[str] = set()
        count = 0
        for store in ("ROOT", "CA"):
            try:
                for cert_bytes, _enc, _trust in ssl.enum_certificates(store):
                    try:
                        pem = ssl.DER_cert_to_PEM_cert(cert_bytes).strip()
                    except Exception:
                        continue
                    if pem not in seen:
                        seen.add(pem)
                        parts.append(pem)
                        count += 1
            except Exception as e:
                logger.debug(f"Tidak bisa membaca store {store}: {e}")
        logger.info(f"CA bundle: menambahkan {count} sertifikat dari Windows store.")

    try:
        _BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BUNDLE_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
        return _BUNDLE_PATH
    except Exception as e:
        logger.warning(f"Gagal menulis CA bundle: {e}")
        return None


def configure_ssl(force_rebuild: bool = False) -> None:
    """
    Konfigurasikan environment agar memakai CA bundle gabungan.
    Idempoten: aman dipanggil berkali-kali.
    """
    bundle = _BUNDLE_PATH
    if force_rebuild or not bundle.exists():
        built = _build_bundle()
        if built is None:
            logger.warning("CA bundle tidak dibangun; pakai default sistem.")
            return
        bundle = built

    bundle_str = str(bundle)
    # Hanya set jika belum diset eksplisit oleh user (hormati override manual)
    os.environ.setdefault("CURL_CA_BUNDLE", bundle_str)
    os.environ.setdefault("SSL_CERT_FILE", bundle_str)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle_str)
    logger.info(f"SSL CA bundle aktif: {bundle_str}")


# Jalankan otomatis saat modul diimpor
configure_ssl()

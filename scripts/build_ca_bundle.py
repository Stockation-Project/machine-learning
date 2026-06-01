"""
build_ca_bundle.py
==================
Membangun CA bundle gabungan (certifi + root CA dari Windows cert store)
agar yfinance/curl_cffi dan requests bisa memverifikasi TLS di lingkungan
yang memakai antivirus/proxy penyadap HTTPS.

Output: artifacts/system_ca_bundle.pem

Pakai dengan menyetel environment variable:
    CURL_CA_BUNDLE  -> dipakai libcurl (curl_cffi/yfinance)
    SSL_CERT_FILE   -> dipakai ssl/requests bawaan Python
"""
import ssl
import subprocess
from pathlib import Path

import certifi

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "artifacts" / "system_ca_bundle.pem"


def get_windows_ca_pems() -> list[str]:
    """Ambil sertifikat dari Windows cert store (ROOT + CA) sebagai PEM."""
    pems: set[str] = set()
    try:
        for store in ("ROOT", "CA"):
            for cert_bytes, encoding, trust in ssl.enum_certificates(store):
                try:
                    pem = ssl.DER_cert_to_PEM_cert(cert_bytes)
                    pems.add(pem.strip())
                except Exception:
                    continue
    except Exception as e:
        print(f"WARN: gagal membaca Windows store via ssl: {e}")
    return list(pems)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # 1. Mulai dari bundle certifi (CA publik standar)
    base = Path(certifi.where()).read_text(encoding="utf-8")
    parts = [base.strip()]

    # 2. Tambahkan CA dari Windows store (termasuk root CA antivirus/proxy)
    win_pems = get_windows_ca_pems()
    print(f"Menambahkan {len(win_pems)} sertifikat dari Windows store.")
    parts.extend(win_pems)

    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"CA bundle ditulis ke: {OUT}")
    print(f"Ukuran: {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()

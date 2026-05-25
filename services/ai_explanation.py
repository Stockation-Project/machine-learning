import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Konfigurasi Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class AIExplanationService:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY tidak ditemukan di environment variables.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def explain_term(self, term: str, context: str = None) -> str:
        prompt = f"Jelaskan istilah saham berikut secara singkat (maksimal 2-3 kalimat) dan menggunakan bahasa yang sangat mudah dipahami oleh orang awam yang baru belajar investasi saham: {term}."
        if context:
            prompt += f"\n\nKonteks tambahan: {context}. Masukkan konteks spesifik ini ke dalam penjelasanmu."
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Gagal mendapatkan penjelasan dari AI: {str(e)}")

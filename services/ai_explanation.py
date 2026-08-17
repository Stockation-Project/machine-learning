import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Konfigurasi Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class AIExplanationService:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY tidak ditemukan di environment variables.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = 'openai/gpt-oss-120b'

    def explain_term(self, term: str, context: str = None) -> str:
        prompt = f"Jelaskan istilah saham berikut secara singkat (maksimal 2-3 kalimat) dan menggunakan bahasa yang sangat mudah dipahami oleh orang awam yang baru belajar investasi saham: {term}."
        if context:
            prompt += f"\n\nKonteks tambahan: {context}. Masukkan konteks spesifik ini ke dalam penjelasanmu."
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=256
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower() or "quota" in err_msg.lower():
                raise Exception("Limit layanan AI sedang habis karena terlalu banyak permintaan. Silakan coba lagi dalam beberapa menit.")
            raise Exception(f"Gagal mendapatkan penjelasan dari AI: {err_msg}")

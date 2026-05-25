import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

working_models = []

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        model_name = m.name.replace('models/', '')
        print(f"Testing {model_name}...")
        model = genai.GenerativeModel(model_name)
        try:
            res = model.generate_content("Hello")
            print(f"SUCCESS: {model_name}")
            working_models.append(model_name)
            break # Just need one working model
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                print(f"FAILED (Quota/Limit 0): {model_name}")
            else:
                print(f"FAILED (Other): {model_name} - {msg.splitlines()[-1] if msg.splitlines() else msg}")

print("\n--- Summary ---")
if working_models:
    print(f"Working model found: {working_models[0]}")
else:
    print("NO WORKING MODELS FOUND. All returned quota limit 0 or errors.")

from services.ai_explanation import AIExplanationService

try:
    svc = AIExplanationService()
    print("Init success!")
    res = svc.explain_term("CAGR")
    print("Result:", res)
except Exception as e:
    import traceback
    traceback.print_exc()

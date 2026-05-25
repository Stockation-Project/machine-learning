import requests

res = requests.post("http://localhost:8000/api/ai/explain", json={
    "term": "Anomali Harga Saham",
    "context": "Saham BREN mengalami anomali dengan pergerakan harga Naik Signifikan."
})

print(res.status_code)
print(res.json())

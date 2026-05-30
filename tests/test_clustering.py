from services.clustering_service import ClusteringService
import json

try:
    print("Starting clustering...")
    service = ClusteringService()
    mapping = service.classify_all_stocks()
    print("Clustering successful!")
    print(json.dumps(mapping, indent=2))
except Exception as e:
    print(f"Error: {e}")

"""
Simulates what GCS sends to Pub/Sub when a CSV lands in raw/.
Use this for local testing with the Pub/Sub emulator.
Run: python simulate_gcs_upload.py
"""
import base64, json, requests

# This is exactly what GCS sends to Pub/Sub on OBJECT_FINALIZE
gcs_event = {
    "bucket":    "vehicle-logs-sachin",
    "name":      "raw/fleet_test_2024.csv",     # the uploaded file
    "eventType": "OBJECT_FINALIZE",
    "contentType": "text/csv",
    "size":      "1024",
}

# Pub/Sub wraps it in base64
payload = {
    "message": {
        "data":        base64.b64encode(json.dumps(gcs_event).encode()).decode(),
        "messageId":   "test-123",
        "publishTime": "2024-01-01T00:00:00Z",
    },
    "subscription": "projects/vehicle-diagnostics-491610/subscriptions/vehicle-csv-sub"
}

# Send to your local API (running via docker-compose)
response = requests.post(
    "http://localhost:8080/etl/trigger",
    json=payload,
    headers={"Content-Type": "application/json"}
)

print(f"Status : {response.status_code}")
print(f"Response: {response.json()}")

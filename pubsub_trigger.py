"""
Pub/Sub Event-Driven ETL Trigger
=================================
Adds automatic ETL execution whenever a CSV lands in GCS raw/.

HOW IT WORKS:
  1. You upload a CSV to gs://vehicle-logs-sachin/raw/
  2. GCS fires a "object finalized" notification → Pub/Sub topic
  3. Pub/Sub push subscription → POST /etl/trigger on your Cloud Run API
  4. This module handles that POST, validates the message, runs ETL

SETUP (one-time gcloud commands — run these once):
  See bottom of this file under "SETUP COMMANDS"
"""

import base64
import json
import logging
import os
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from etl import run_etl

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Pub/Sub push message schema ──────────────────────────────────────────────
# When GCS fires a notification, Pub/Sub wraps it like this:
# {
#   "message": {
#     "data": "<base64 encoded JSON>",   ← the actual GCS event
#     "messageId": "...",
#     "publishTime": "..."
#   },
#   "subscription": "projects/.../subscriptions/..."
# }

class PubSubMessage(BaseModel):
    data: str           # base64-encoded GCS notification JSON
    messageId: str
    publishTime: str

class PubSubPushPayload(BaseModel):
    message: PubSubMessage
    subscription: str


# ── The trigger endpoint ──────────────────────────────────────────────────────
@router.post("/etl/trigger")
async def etl_trigger(request: Request):
    """
    Called automatically by Pub/Sub push subscription when a CSV lands in GCS.
    No manual intervention needed after one-time setup.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Decode the Pub/Sub message
    try:
        message_data = body.get("message", {}).get("data", "")
        decoded      = base64.b64decode(message_data).decode("utf-8")
        gcs_event    = json.loads(decoded)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode message: {e}")

    # GCS notification fields
    bucket_id   = gcs_event.get("bucket", "")
    object_name = gcs_event.get("name", "")       # e.g. "raw/fleet_2024_01.csv"
    event_type  = gcs_event.get("eventType", "")   # "OBJECT_FINALIZE"

    logger.info(f"Pub/Sub event: {event_type} | gs://{bucket_id}/{object_name}")

    # Only process CSV files landing in raw/
    if not object_name.startswith("raw/") or not object_name.endswith(".csv"):
        logger.info(f"Skipping non-CSV or non-raw/ file: {object_name}")
        # Return 200 so Pub/Sub doesn't retry — this is intentional
        return {"status": "skipped", "reason": "not a raw CSV file"}

    # Only process on finalization (not on delete/metadata update)
    if event_type and event_type != "OBJECT_FINALIZE":
        return {"status": "skipped", "reason": f"event type {event_type} ignored"}

    # Run ETL
    logger.info(f"Triggering ETL for: {object_name}")
    try:
        run_etl()
        return {"status": "success", "file": object_name}
    except Exception as e:
        logger.error(f"ETL failed for {object_name}: {e}")
        # Return 500 → Pub/Sub will retry automatically (up to your retry policy)
        raise HTTPException(status_code=500, detail=f"ETL failed: {e}")


"""
══════════════════════════════════════════════════════════════════
SETUP COMMANDS — run these ONCE to wire up the automation
══════════════════════════════════════════════════════════════════

Step 1: Create the Pub/Sub topic
─────────────────────────────────
gcloud pubsub topics create vehicle-csv-uploads \
    --project=vehicle-diagnostics-491610

Step 2: Tell GCS to publish to this topic when a CSV lands in raw/
────────────────────────────────────────────────────────────────────
gcloud storage buckets notifications create gs://vehicle-logs-sachin \
    --topic=vehicle-csv-uploads \
    --event-types=OBJECT_FINALIZE \
    --object-prefix=raw/ \
    --payload-format=json \
    --project=vehicle-diagnostics-491610

Step 3: Create a push subscription pointing to your Cloud Run API
──────────────────────────────────────────────────────────────────
gcloud pubsub subscriptions create vehicle-csv-sub \
    --topic=vehicle-csv-uploads \
    --push-endpoint=https://vehicle-diagnostics-api-746368083201.asia-south1.run.app/etl/trigger \
    --ack-deadline=300 \
    --min-retry-delay=10s \
    --max-retry-delay=300s \
    --project=vehicle-diagnostics-491610

Step 4: Grant Pub/Sub permission to invoke your Cloud Run service
──────────────────────────────────────────────────────────────────
gcloud run services add-iam-policy-binding vehicle-diagnostics-api \
    --region=asia-south1 \
    --member=serviceAccount:service-746368083201@gcp-sa-pubsub.iam.gserviceaccount.com \
    --role=roles/run.invoker \
    --project=vehicle-diagnostics-491610

After these 4 commands:
  Upload ANY CSV to gs://vehicle-logs-sachin/raw/  →  ETL runs automatically.
  No manual python etl.py needed ever again.

══════════════════════════════════════════════════════════════════
LOCAL TESTING with Pub/Sub emulator (docker-compose)
══════════════════════════════════════════════════════════════════

Your docker-compose.yml already has the emulator on port 8085.

# Terminal 1 — start the stack
docker-compose up

# Terminal 2 — create topic + subscription in emulator
export PUBSUB_EMULATOR_HOST=localhost:8085
python3 -c "
from google.cloud import pubsub_v1
publisher  = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()

topic_path = publisher.topic_path('vehicle-diagnostics-491610', 'vehicle-csv-uploads')
sub_path   = subscriber.subscription_path('vehicle-diagnostics-491610', 'vehicle-csv-sub')

publisher.create_topic(request={'name': topic_path})
subscriber.create_subscription(request={
    'name': sub_path,
    'topic': topic_path,
    'push_config': {'push_endpoint': 'http://localhost:8080/etl/trigger'},
    'ack_deadline_seconds': 300,
})
print('Topic and subscription created in emulator')
"

# Terminal 3 — simulate a GCS notification (as if a CSV was uploaded)
python3 simulate_gcs_upload.py
"""

from google.cloud import storage, bigquery, aiplatform
from google.cloud.aiplatform_v1.types import IndexDatapoint
from vertexai.language_models import TextEmbeddingModel
import pandas as pd, io, logging, datetime
from config import PROJECT_ID, REGION, BUCKET, BQ_TABLE, EMBEDDING_MODEL, VS_INDEX_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
aiplatform.init(project=PROJECT_ID, location=REGION)


def read_raw_files():
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blobs  = [b for b in bucket.list_blobs(prefix="raw/") if b.name.endswith(".csv")]
    logging.info(f"Found {len(blobs)} CSV files")
    return blobs, bucket


def clean(raw_text: str, filename: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(raw_text))
    df["timestamp"]    = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["processed_at"] = datetime.datetime.utcnow()
    df["source_file"]  = filename
    return df.dropna(subset=["timestamp"])


def load_to_bigquery(df: pd.DataFrame) -> None:
    if df.empty: return
    client = bigquery.Client()
    df["value"] = pd.to_numeric(df.get("value", pd.Series()), errors="coerce")
    job = client.load_table_from_dataframe(df, BQ_TABLE, job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("timestamp",    "TIMESTAMP"),
            bigquery.SchemaField("vehicle_id",   "STRING"),
            bigquery.SchemaField("error_code",   "STRING"),
            bigquery.SchemaField("sensor",       "STRING"),
            bigquery.SchemaField("value",        "FLOAT64"),
            bigquery.SchemaField("unit",         "STRING"),
            bigquery.SchemaField("processed_at", "TIMESTAMP"),
            bigquery.SchemaField("source_file",  "STRING"),
        ]))
    job.result()
    logging.info(f"Loaded {len(df)} rows into BigQuery")


def embed_and_index(df: pd.DataFrame) -> None:
    if df.empty or VS_INDEX_ID == "": return
    errors = df.dropna(subset=["error_code"]).copy()
    if errors.empty: return
    errors["description"] = errors.apply(
        lambda r: f"Vehicle {r['vehicle_id']} reported OBD-II fault {r['error_code']} "
                  f"on {r['sensor']} sensor reading {r['value']} {r.get('unit','')}.", axis=1)
    model      = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    embeddings = model.get_embeddings(errors["description"].tolist())
    datapoints = [
        IndexDatapoint(
            datapoint_id=f"{r['vehicle_id']}_{r['error_code']}",
            feature_vector=embeddings[i].values,
        )
        for i, (_, r) in enumerate(errors.iterrows())
    ]
    index = aiplatform.MatchingEngineIndex(VS_INDEX_ID)
    index.upsert_datapoints(datapoints=datapoints)
    logging.info(f"Upserted {len(datapoints)} vectors to Vertex AI Vector Search")


def run_etl() -> None:
    logging.info("=== ETL started ===")
    blobs, bucket = read_raw_files()
    if not blobs:
        logging.info("Nothing to process."); return
    all_dfs = []
    for blob in blobs:
        df = clean(blob.download_as_text(), blob.name)
        bucket.blob(blob.name.replace("raw/","processed/")).upload_from_string(
            df.to_csv(index=False), content_type="text/csv")
        blob.delete()
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    load_to_bigquery(combined)
    embed_and_index(combined)
    logging.info(f"=== ETL complete: {len(combined)} rows ===")


if __name__ == "__main__":
    run_etl()

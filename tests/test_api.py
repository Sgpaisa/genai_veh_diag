import sys
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api import app
from diagnostics import DiagnosisResult

http = TestClient(app)


class _BQRow:
    """Minimal stand-in for a BigQuery result row."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── GET /health ────────────────────────────────────────────────────────

def test_health_ok():
    resp = http.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "2.0"


# ── GET /vehicle/{vehicle_id} ──────────────────────────────────────────

def test_vehicle_errors_returns_records():
    rows = [
        _BQRow(error_code="P0171", sensor="oxygen_sensor",
               value=0.12, timestamp="2024-01-01 00:00:00"),
    ]
    with patch("api.bq") as mock_bq:
        mock_bq.query.return_value.result.return_value = rows
        resp = http.get("/vehicle/VEH101")

    assert resp.status_code == 200
    body = resp.json()
    assert body["vehicle_id"] == "VEH101"
    assert body["total_errors"] == 1
    assert body["errors"][0]["error_code"] == "P0171"
    assert body["errors"][0]["sensor"] == "oxygen_sensor"


def test_vehicle_errors_404_when_no_data():
    with patch("api.bq") as mock_bq:
        mock_bq.query.return_value.result.return_value = []
        resp = http.get("/vehicle/UNKNOWN")

    assert resp.status_code == 404


# ── GET /fleet/summary ─────────────────────────────────────────────────

def test_fleet_summary_returns_top_errors():
    rows = [
        _BQRow(error_code="P0171", occ=42, vehicles=7),
        _BQRow(error_code="P0300", occ=15, vehicles=3),
    ]
    with patch("api.bq") as mock_bq:
        mock_bq.query.return_value.result.return_value = rows
        resp = http.get("/fleet/summary")

    assert resp.status_code == 200
    entries = resp.json()["fleet_top_errors"]
    assert len(entries) == 2
    assert entries[0]["error_code"] == "P0171"
    assert entries[0]["occurrences"] == 42
    assert entries[0]["vehicles_affected"] == 7


# ── GET /diagnose/{vehicle_id}/{error_code} ────────────────────────────

def test_diagnose_route_returns_structured_response():
    fake_result = DiagnosisResult(
        risk_level="High",
        root_causes=["fuel system fault"],
        immediate_actions=["check fuel pressure"],
        summary="High risk detected",
    )
    with patch("api.diagnose", return_value=fake_result):
        resp = http.get("/diagnose/VEH101/P0171?sensor=oxygen_sensor&value=0.12V")

    assert resp.status_code == 200
    body = resp.json()
    assert body["vehicle_id"] == "VEH101"
    assert body["error_code"] == "P0171"
    assert body["risk_level"] == "High"
    assert "fuel system fault" in body["root_causes"]
    assert body["summary"] == "High risk detected"


# ── GET /search ────────────────────────────────────────────────────────

def test_search_503_when_endpoint_not_configured():
    with patch("api.VS_ENDPOINT_ID", ""):
        resp = http.get("/search?symptom=engine+knocking")
    assert resp.status_code == 503


def test_search_returns_neighbor_results():
    fake_neighbor = MagicMock()
    fake_neighbor.distance = 0.87
    fake_neighbor.id = "VEH101_P0171"

    # TextEmbeddingModel is imported inside the route at call time;
    # configure it through the sys.modules mock set up in conftest.py.
    lm_mod = sys.modules["vertexai.language_models"]
    fake_model = MagicMock()
    fake_model.get_embeddings.return_value = [MagicMock(values=[0.1] * 5)]
    lm_mod.TextEmbeddingModel.from_pretrained.return_value = fake_model

    fake_endpoint = MagicMock()
    fake_endpoint.find_neighbors.return_value = [[fake_neighbor]]

    with patch("api.VS_ENDPOINT_ID", "some-endpoint-id"), \
         patch("api.aiplatform") as mock_aip:
        mock_aip.MatchingEngineIndexEndpoint.return_value = fake_endpoint
        resp = http.get("/search?symptom=engine+knocking")

    assert resp.status_code == 200
    body = resp.json()
    assert body["query_symptom"] == "engine knocking"
    results = body["results"]
    assert len(results) == 1
    assert results[0]["rank"] == 1
    assert results[0]["vehicle_id"] == "VEH101"
    assert results[0]["error_code"] == "P0171"
    assert results[0]["similarity_pct"] == 87.0

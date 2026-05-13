from unittest.mock import MagicMock, patch

from diagnostics import diagnose, DiagnosisResult


class _BQRow:
    """Minimal stand-in for a BigQuery row with an error_code attribute."""
    def __init__(self, error_code: str):
        self.error_code = error_code


def _make_result(risk="Low", causes=None, actions=None, summary="all clear"):
    return DiagnosisResult(
        risk_level=risk,
        root_causes=causes or ["lean fuel mixture"],
        immediate_actions=actions or ["check fuel injectors"],
        summary=summary,
    )


# ── diagnose() happy path ──────────────────────────────────────────────

def test_diagnose_returns_diagnosis_result():
    expected = _make_result(risk="Low")
    mock_resp = MagicMock()
    mock_resp.parsed = expected

    with patch("diagnostics.bigquery.Client") as mock_bq, \
         patch("diagnostics.client") as mock_genai:
        mock_bq.return_value.query.return_value.result.return_value = [
            _BQRow("P0300")
        ]
        mock_genai.models.generate_content.return_value = mock_resp

        result = diagnose("VEH101", "P0171", "oxygen_sensor", "0.12V")

    assert isinstance(result, DiagnosisResult)
    assert result.risk_level == "Low"
    assert result.root_causes == ["lean fuel mixture"]
    assert result.immediate_actions == ["check fuel injectors"]


def test_diagnose_high_risk():
    expected = _make_result(
        risk="High",
        causes=["fuel leak"],
        actions=["stop vehicle immediately"],
    )
    mock_resp = MagicMock()
    mock_resp.parsed = expected

    with patch("diagnostics.bigquery.Client") as mock_bq, \
         patch("diagnostics.client") as mock_genai:
        mock_bq.return_value.query.return_value.result.return_value = []
        mock_genai.models.generate_content.return_value = mock_resp

        result = diagnose("VEH999", "P0300", "injector", "0.0V")

    assert result.risk_level == "High"
    assert "stop vehicle immediately" in result.immediate_actions


# ── prompt construction ────────────────────────────────────────────────

def test_diagnose_empty_history_uses_none_in_prompt():
    mock_resp = MagicMock()
    mock_resp.parsed = _make_result()

    with patch("diagnostics.bigquery.Client") as mock_bq, \
         patch("diagnostics.client") as mock_genai:
        mock_bq.return_value.query.return_value.result.return_value = []
        mock_genai.models.generate_content.return_value = mock_resp

        diagnose("VEH101", "P0171", "oxygen_sensor", "0.12V")

        prompt = mock_genai.models.generate_content.call_args.kwargs["contents"]

    assert "Recent error history: none" in prompt


def test_diagnose_includes_history_codes_in_prompt():
    mock_resp = MagicMock()
    mock_resp.parsed = _make_result()

    with patch("diagnostics.bigquery.Client") as mock_bq, \
         patch("diagnostics.client") as mock_genai:
        mock_bq.return_value.query.return_value.result.return_value = [
            _BQRow("P0420"),
            _BQRow("P0442"),
        ]
        mock_genai.models.generate_content.return_value = mock_resp

        diagnose("VEH101", "P0171", "oxygen_sensor", "0.12V")

        prompt = mock_genai.models.generate_content.call_args.kwargs["contents"]

    assert "P0420" in prompt
    assert "P0442" in prompt

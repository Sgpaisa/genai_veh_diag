"""
Inject mock GCP modules into sys.modules before any project code is imported.
This lets diagnostics.py and api.py load without real credentials or network access.
"""
import sys
from unittest.mock import MagicMock

# Force-replace every GCP leaf module we need to intercept.
_MOCKS = {
    "google.cloud.bigquery": MagicMock(),
    "google.cloud.aiplatform": MagicMock(),
    "google.genai": MagicMock(),
    "google.genai.types": MagicMock(),
    "vertexai": MagicMock(),
    "vertexai.language_models": MagicMock(),
}
for _name, _mock in _MOCKS.items():
    sys.modules[_name] = _mock

# Add parent namespace packages only if they are not already present
# (they may be real namespace packages on the host).
for _parent in ("google", "google.cloud", "google.auth"):
    if _parent not in sys.modules:
        sys.modules[_parent] = MagicMock()

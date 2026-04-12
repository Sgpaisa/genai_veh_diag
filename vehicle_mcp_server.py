import sys
import logging

# CRITICAL: redirect all logging to stderr, never stdout
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from mcp.server.fastmcp import FastMCP
from google.cloud import bigquery
from config import BQ_TABLE, PROJECT_ID

mcp = FastMCP("Vehicle Diagnostics MCP Server")
bq  = bigquery.Client()


@mcp.tool()
def get_vehicle_errors(vehicle_id: str) -> dict:
    """Get all OBD-II error codes for a specific vehicle ID from BigQuery.
    Use this when the user asks about a specific vehicle's error history."""
    rows = bq.query(
        f"SELECT error_code, sensor, value, timestamp FROM `{BQ_TABLE}` "
        f"WHERE vehicle_id = '{vehicle_id}' ORDER BY timestamp DESC LIMIT 20"
    ).result()
    return {
        "vehicle_id": vehicle_id,
        "errors": [
            {"code": r.error_code, "sensor": r.sensor, "value": r.value}
            for r in rows
        ]
    }


@mcp.tool()
def get_fleet_summary() -> dict:
    """Get a summary of all vehicles and their error counts from BigQuery.
    Use this when the user asks about fleet health or overall diagnostics."""
    rows = bq.query(
        f"SELECT vehicle_id, COUNT(*) as error_count "
        f"FROM `{BQ_TABLE}` "
        f"GROUP BY vehicle_id ORDER BY error_count DESC LIMIT 20"
    ).result()
    return {
        "fleet_summary": [
            {"vehicle_id": r.vehicle_id, "error_count": r.error_count}
            for r in rows
        ]
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

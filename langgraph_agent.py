from langgraph.graph import StateGraph, END
from typing import TypedDict
from diagnostics import diagnose
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class FleetState(TypedDict):
    vehicle_id:  str
    error_code:  str
    sensor:      str
    value:       str
    risk_level:  str    # filled by diagnose_node
    alert_sent:  bool   # filled by alert_node
    report:      str    # filled by final nodes


def diagnose_node(state: FleetState) -> FleetState:
    """Call Gemini to diagnose the vehicle error."""
    result = diagnose(state["vehicle_id"], state["error_code"],
                      state["sensor"], state["value"])
    return {**state, "risk_level": result.risk_level}


def alert_node(state: FleetState) -> FleetState:
    """Send URGENT alert for High-risk diagnosis."""
    logging.warning(f"ALERT: {state['vehicle_id']}/{state['error_code']} is HIGH risk!")
    return {**state, "alert_sent": True,
            "report": f"URGENT: {state['vehicle_id']} requires immediate inspection."}


def monitor_node(state: FleetState) -> FleetState:
    """Log for Medium/Low risk — no alert needed."""
    logging.info(f"MONITORING: {state['vehicle_id']}/{state['error_code']} risk={state['risk_level']}")
    return {**state, "alert_sent": False,
            "report": f"MONITORING: {state['vehicle_id']} added to watch list."}


def route_by_risk(state: FleetState) -> str:
    """Conditional edge — route to alert or monitor based on risk_level."""
    return "alert" if state["risk_level"] == "High" else "monitor"


# Build the StateGraph
graph = StateGraph(FleetState)
graph.add_node("diagnose", diagnose_node)
graph.add_node("alert",    alert_node)
graph.add_node("monitor",  monitor_node)
graph.set_entry_point("diagnose")
graph.add_conditional_edges("diagnose", route_by_risk,
    {"alert": "alert", "monitor": "monitor"})
graph.add_edge("alert",   END)
graph.add_edge("monitor", END)
app = graph.compile()


if __name__ == "__main__":
    # Test with a High-risk vehicle error
    result = app.invoke({
        "vehicle_id": "VEH101", "error_code": "P0171",
        "sensor": "oxygen_sensor", "value": "0.12V",
        "risk_level": "", "alert_sent": False, "report": ""
    })
    print(f"Final report: {result['report']}")
    print(f"Alert sent:   {result['alert_sent']}")

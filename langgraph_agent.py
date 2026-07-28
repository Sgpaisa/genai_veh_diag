"""
LangGraph Fleet Diagnostic Agent — with Human-in-the-Loop Approval
====================================================================

Graph structure:
                    ┌─────────┐
                    │ diagnose│
                    └────┬────┘
                         │
              conditional edge (route_by_risk)
                 /                    \
         risk==High              risk==Medium/Low
               /                          \
  ┌────────────────┐               ┌──────────────┐
  │ await_approval │  ← PAUSES     │   monitor    │
  │ (human reads)  │    HERE       └──────┬───────┘
  └────────┬───────┘                      │
           │  human approves              END
           ▼
      ┌──────────┐
      │  alert   │
      └────┬─────┘
           │
          END

Key concepts implemented:
  - interrupt_before=["await_approval"] → graph PAUSES before this node
  - MemorySaver checkpointer → full state saved to memory when paused
  - graph.invoke() resumes from saved state when human approves
  - thread_id → each vehicle diagnosis is an independent thread
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional
from diagnostics import diagnose
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── State: the data bag that flows through every node ────────────────────────
# Every field must be defined here. Nodes read from state and return updates.

class FleetState(TypedDict):
    vehicle_id:       str
    error_code:       str
    sensor:           str
    value:            str
    risk_level:       str            # filled by diagnose_node
    root_causes:      list[str]      # filled by diagnose_node
    immediate_actions: list[str]     # filled by diagnose_node
    summary:          str            # filled by diagnose_node
    approved:         bool           # set to True by human before resuming
    alert_sent:       bool           # filled by alert_node
    report:           str            # filled by alert_node or monitor_node


# ── Node 1: diagnose ─────────────────────────────────────────────────────────
# Calls Gemini via diagnostics.py. Fills risk_level, root_causes, actions, summary.
# Graph always starts here.

def diagnose_node(state: FleetState) -> FleetState:
    """Call Gemini to diagnose the vehicle error. Fills risk_level and diagnosis details."""
    logging.info(f"[diagnose] Running diagnosis for {state['vehicle_id']}/{state['error_code']}")

    result = diagnose(
        state["vehicle_id"],
        state["error_code"],
        state["sensor"],
        state["value"]
    )

    logging.info(f"[diagnose] Result: risk_level={result.risk_level} for {state['vehicle_id']}")

    return {
        **state,
        "risk_level":        result.risk_level,
        "root_causes":       result.root_causes,
        "immediate_actions": result.immediate_actions,
        "summary":           result.summary,
    }


# ── Node 2: await_approval ───────────────────────────────────────────────────
# This node does NOTHING by itself.
# Its only purpose is to be the interrupt point.
# graph.compile(interrupt_before=["await_approval"]) makes the graph
# PAUSE before entering this node and save state to MemorySaver.
# The human reads the diagnosis, then resumes by calling graph.invoke()
# again with the same thread_id and approved=True in the state.

def await_approval_node(state: FleetState) -> FleetState:
    """
    Pause point for human review. Graph resumes only after human approves.
    This node only runs AFTER the human has approved — it just logs the approval.
    """
    logging.info(f"[await_approval] Human approved alert for {state['vehicle_id']}")
    return state   # pass state through unchanged — alert_node does the real work


# ── Node 3: alert ─────────────────────────────────────────────────────────────
# Only reached after human approves. Sends the URGENT alert.

def alert_node(state: FleetState) -> FleetState:
    """Send URGENT alert — only runs after human approval via await_approval."""
    logging.warning(
        f"[alert] URGENT ALERT SENT: {state['vehicle_id']}/{state['error_code']}"
        f" | Risk: {state['risk_level']}"
        f" | Actions: {state['immediate_actions']}"
    )
    report = (
        f"URGENT [{state['vehicle_id']}] — {state['error_code']}\n"
        f"Risk     : {state['risk_level']}\n"
        f"Summary  : {state['summary']}\n"
        f"Causes   : {', '.join(state['root_causes'])}\n"
        f"Actions  : {', '.join(state['immediate_actions'])}\n"
        f"Status   : Alert sent — inspect within 4 hours."
    )
    return {**state, "alert_sent": True, "report": report}


# ── Node 4: monitor ───────────────────────────────────────────────────────────
# For Medium/Low risk — no alert, no approval needed. Straight through.

def monitor_node(state: FleetState) -> FleetState:
    """Log Medium/Low risk vehicle for monitoring. No human approval needed."""
    logging.info(
        f"[monitor] {state['vehicle_id']}/{state['error_code']}"
        f" risk={state['risk_level']} — added to watch list"
    )
    report = (
        f"MONITORING [{state['vehicle_id']}] — {state['error_code']}\n"
        f"Risk    : {state['risk_level']}\n"
        f"Summary : {state['summary']}\n"
        f"Action  : Added to watch list. No immediate action required."
    )
    return {**state, "alert_sent": False, "report": report}


# ── Conditional edge function ─────────────────────────────────────────────────
# Called after diagnose_node. Returns the name of the NEXT node.
# High risk → await_approval (which will be interrupted before execution)
# Anything else → monitor (runs straight through, no interrupt)

def route_by_risk(state: FleetState) -> str:
    """Route High risk to human approval path. Medium/Low go straight to monitor."""
    if state["risk_level"] == "High":
        return "await_approval"
    return "monitor"


# ── Build the graph ───────────────────────────────────────────────────────────

checkpointer = MemorySaver()
# MemorySaver stores the full state in memory when the graph pauses.
# In production, swap this for SqliteSaver or RedisSaver so state
# survives server restarts and can be resumed hours later.

graph = StateGraph(FleetState)

# Register all nodes
graph.add_node("diagnose",        diagnose_node)
graph.add_node("await_approval",  await_approval_node)
graph.add_node("alert",           alert_node)
graph.add_node("monitor",         monitor_node)

# Entry point — always starts at diagnose
graph.set_entry_point("diagnose")

# After diagnose: conditional branch based on risk_level
graph.add_conditional_edges(
    "diagnose",
    route_by_risk,
    {
        "await_approval": "await_approval",   # High risk path
        "monitor":        "monitor",           # Medium/Low path
    }
)

# After await_approval → alert (only reached after human resumes)
graph.add_edge("await_approval", "alert")

# Terminal edges
graph.add_edge("alert",   END)
graph.add_edge("monitor", END)

# compile() with:
#   checkpointer  → saves state to MemorySaver when graph pauses
#   interrupt_before=["await_approval"] → PAUSE before entering await_approval
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["await_approval"]
)


# ── Runner with human-in-the-loop ────────────────────────────────────────────

def run_with_approval(vehicle_id: str, error_code: str,
                       sensor: str, value: str) -> dict:
    """
    Full human-in-the-loop flow:
      1. Run graph → pauses if High risk (before await_approval node)
      2. Show human the diagnosis
      3. Human approves or rejects
      4. If approved → resume graph → alert sent
      5. If rejected → return without alert
    """
    # Each vehicle diagnosis is an independent thread.
    # thread_id lets MemorySaver store and retrieve the right state.
    thread_id = f"{vehicle_id}_{error_code}"
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state: FleetState = {
        "vehicle_id":        vehicle_id,
        "error_code":        error_code,
        "sensor":            sensor,
        "value":             value,
        "risk_level":        "",
        "root_causes":       [],
        "immediate_actions": [],
        "summary":           "",
        "approved":          False,
        "alert_sent":        False,
        "report":            "",
    }

    # ── Step 1: Run until interrupt ──────────────────────────────────────────
    # For High risk: graph runs diagnose_node, hits interrupt_before,
    # saves state to MemorySaver, returns the current (partial) state.
    # For Medium/Low: graph runs all the way to END with no interruption.

    print(f"\n{'='*60}")
    print(f"Running diagnosis: {vehicle_id} / {error_code}")
    print(f"{'='*60}")

    state_after_diagnose = app.invoke(initial_state, config=config)

    # Check if graph finished completely (Medium/Low risk — no interrupt)
    if state_after_diagnose.get("risk_level") != "High":
        print(f"\nRisk level: {state_after_diagnose['risk_level']}")
        print(f"Report:\n{state_after_diagnose['report']}")
        return state_after_diagnose

    # ── Step 2: Graph paused — show human the diagnosis ──────────────────────
    print(f"\n[!] HIGH RISK DETECTED -- Human approval required")
    print(f"{'-'*60}")
    print(f"Vehicle  : {state_after_diagnose['vehicle_id']}")
    print(f"Error    : {state_after_diagnose['error_code']}")
    print(f"Risk     : {state_after_diagnose['risk_level']}")
    print(f"Summary  : {state_after_diagnose['summary']}")
    print(f"Causes   : {', '.join(state_after_diagnose['root_causes'])}")
    print(f"Actions  : {', '.join(state_after_diagnose['immediate_actions'])}")
    print(f"{'-'*60}")

    # ── Step 3: Human decision ────────────────────────────────────────────────
    decision = input("Send alert? (yes/no): ").strip().lower()

    if decision != "yes":
        print("Alert rejected by operator. Vehicle added to watch list.")
        return {**state_after_diagnose, "alert_sent": False,
                "report": f"Alert REJECTED by operator for {vehicle_id}/{error_code}"}

    # ── Step 4: Resume graph ──────────────────────────────────────────────────
    # Calling invoke() again with the SAME thread_id resumes from the saved
    # checkpoint — it picks up right where it paused (before await_approval).
    # We update approved=True in the state so await_approval_node can log it.

    print("\nResuming graph — sending alert...")
    final_state = app.invoke(
        {"approved": True},    # only send the updated field
        config=config          # SAME thread_id → resumes saved state
    )

    print(f"\nReport:\n{final_state['report']}")
    return final_state


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Test 1: High risk — will pause and ask for human approval
    print("\n--- TEST 1: High risk vehicle (expects human approval prompt) ---")
    result = run_with_approval(
        vehicle_id="VEH101",
        error_code="P0300",     # Random misfire — typically High risk
        sensor="crankshaft_sensor",
        value="0.0V"
    )
    print(f"\nAlert sent: {result['alert_sent']}")

    # Test 2: Low risk — runs straight through, no approval needed
    print("\n--- TEST 2: Low risk vehicle (no approval needed) ---")
    result = run_with_approval(
        vehicle_id="VEH205",
        error_code="P0442",     # Small EVAP leak — typically Low risk
        sensor="evap_sensor",
        value="0.8V"
    )
    print(f"\nAlert sent: {result['alert_sent']}")

"""
Golden dataset -- the human-labeled ground truth used for PRE-production evaluation.

Every row is a case a domain expert has already graded. This is what you
run a new model/prompt version against BEFORE it ever sees production traffic.

Two kinds of labels live here on purpose, because LLM evals need both:
  - expected_risk   -> a hard label (safety-critical, must match exactly)
  - rubric          -> soft criteria for judging free-text quality
                       (used by judge.py, which mimics "LLM-as-judge")
"""

GOLDEN_SET = [
    {
        "vehicle_id": "VEH101", "error_code": "P0171", "sensor": "oxygen_sensor",
        "value": "0.12V", "history": "P0171, P0171",
        "expected_risk": "High",
        "rubric": ["mentions lean mixture or fuel/air ratio", "recommends inspecting oxygen sensor or fuel system"],
    },
    {
        "vehicle_id": "VEH101", "error_code": "P0128", "sensor": "coolant_temp",
        "value": "68C", "history": "none",
        "expected_risk": "Medium",
        "rubric": ["mentions thermostat", "notes engine running cooler than expected"],
    },
    {
        "vehicle_id": "VEH102", "error_code": "P0300", "sensor": "misfire_count",
        "value": "18", "history": "P0300, P0171",
        "expected_risk": "High",
        "rubric": ["mentions misfire", "recommends urgent inspection or stopping vehicle use"],
    },
    {
        "vehicle_id": "VEH103", "error_code": "P0420", "sensor": "catalyst",
        "value": "0.71", "history": "none",
        "expected_risk": "Medium",
        "rubric": ["mentions catalytic converter efficiency", "does not claim engine is unsafe to drive"],
    },
    {
        "vehicle_id": "VEH102", "error_code": "P0171", "sensor": "oxygen_sensor",
        "value": "0.09V", "history": "P0171",
        "expected_risk": "High",
        "rubric": ["mentions lean mixture or fuel/air ratio", "recommends inspection"],
    },
    {
        "vehicle_id": "VEH104", "error_code": "P0442", "sensor": "evap_system",
        "value": "leak_small", "history": "none",
        "expected_risk": "Low",
        "rubric": ["mentions evaporative emissions or small leak", "does not overstate urgency"],
    },
    {
        "vehicle_id": "VEH105", "error_code": "P0217", "sensor": "coolant_temp",
        "value": "118C", "history": "P0217",
        "expected_risk": "High",
        "rubric": ["mentions engine overheating", "recommends stopping the vehicle immediately"],
    },
    {
        "vehicle_id": "VEH106", "error_code": "P0562", "sensor": "battery_voltage",
        "value": "10.1V", "history": "none",
        "expected_risk": "Medium",
        "rubric": ["mentions low system voltage or battery/alternator", "recommends electrical check"],
    },
    {
        "vehicle_id": "VEH107", "error_code": "P0011", "sensor": "camshaft_timing",
        "value": "over_advanced", "history": "P0011, P0011",
        "expected_risk": "High",
        "rubric": ["mentions variable valve timing or camshaft", "recommends prompt inspection"],
    },
    {
        "vehicle_id": "VEH108", "error_code": "P0455", "sensor": "evap_system",
        "value": "leak_large", "history": "none",
        "expected_risk": "Low",
        "rubric": ["mentions evaporative emissions leak", "notes it is an emissions issue not a drivability issue"],
    },
]

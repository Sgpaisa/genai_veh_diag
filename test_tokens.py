import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
codes = ["P0171", "P0300", "VEH101", "oxygen_sensor", "unbelievable"]
for c in codes:
    tokens = enc.encode(c)
    print(f"{c:20s} → {len(tokens)} tokens: {tokens}")
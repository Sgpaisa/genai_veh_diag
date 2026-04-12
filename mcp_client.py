import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai

gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

async def ask_vehicle_agent(question):
    server = StdioServerParameters(
        command="python3",
        args=["/home/sachin_gattani/vehicle_project/vehicle_mcp_server.py"],
        env={
            "GOOGLE_CLOUD_PROJECT": "vehicle-diagnostics-491610",
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "GEMINI_API_KEY": os.environ["GEMINI_API_KEY"]
        }
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            fleet_result = await session.call_tool("get_fleet_summary", {})
            fleet_data = json.loads(fleet_result.content[0].text)
            top_vehicle = fleet_data["fleet_summary"][0]["vehicle_id"]
            errors_result = await session.call_tool("get_vehicle_errors", {"vehicle_id": top_vehicle})
            errors_data = json.loads(errors_result.content[0].text)
            prompt = f"""You are an expert automotive diagnostic engineer.
Fleet Summary: {json.dumps(fleet_data, indent=2)}
Detailed errors for {top_vehicle}: {json.dumps(errors_data, indent=2)}
Question: {question}
Give a clean professional diagnostic report with:
1. Fleet health overview
2. Most critical vehicle and why
3. What each error code means
4. Recommended workshop actions"""
            response = gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            print("\n" + "="*60)
            print("VEHICLE DIAGNOSTIC REPORT")
            print("="*60)
            print(response.text)

asyncio.run(ask_vehicle_agent("Give me full fleet health summary and which vehicle needs urgent attention?"))

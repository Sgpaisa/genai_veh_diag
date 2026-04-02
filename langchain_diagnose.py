from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

prompt = PromptTemplate.from_template("""You are an expert automotive diagnostic engineer.
Vehicle: {vehicle_id}
OBD-II Error Code: {error_code}
Sensor: {sensor}
Reading: {value}

Classify risk as High / Medium / Low.
List 3 root causes.
List 3 immediate actions.
Give a one-sentence summary.""")

chain = prompt | llm

if __name__ == "__main__":
    test_cases = [
        {"vehicle_id": "VEH101", "error_code": "P0171", "sensor": "oxygen_sensor", "value": "0.12V"},
        {"vehicle_id": "VEH102", "error_code": "P0300", "sensor": "misfire_count", "value": "18"},
        {"vehicle_id": "VEH103", "error_code": "P0128", "sensor": "coolant_temp",  "value": "68C"},
    ]
    for t in test_cases:
        print(f"\nDiagnosing {t['vehicle_id']} / {t['error_code']}...")
        result = chain.invoke(t)
        print(result.content[:300])
        print("--- trace sent to LangSmith ---")

import requests

query = "revenue growth"
num_queries = 3
url = "http://localhost:1234/v1/chat/completions"
prompt = f"""
Generate {num_queries} search queries similar to the input.

Rules:
- Do NOT answer
- Keep queries short
- Use different wording
- Return ONLY valid JSON

Format:
["query1", "query2", "query3"]

Input:
{query}
"""

response = requests.post(
    url,
    json={
        "model": "microsoft/phi-4-mini-reasoning",
        "messages": [
            {"role": "system", "content": "You generate search queries."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 100
    }
)
try:
    data = response.json()
    print("RAW Response:", data.get("choices", [{}])[0].get("message", {}).get("content", data))
except Exception as e:
    print(e)

import requests
import json

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
        "max_tokens": 2048
    }
)
try:
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    print(f"RAW TEXT: {text!r}")
    
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "</reasoning>" in text:
        text = text.split("</reasoning>")[-1].strip()
        
    print(f"POST STRIP TEXT: {text!r}")
    
    queries = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            queries = [q.strip() for q in parsed if isinstance(q, str) and len(q) > 5]
        print(f"JSON LOADS SUCCESS: {queries}")
    except Exception as e:
        print(f"JSON LOADS FAILED: {e}")
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(text[start:end+1])
                if isinstance(parsed, list):
                    queries = [q.strip() for q in parsed if isinstance(q, str) and len(q) > 5]
                print(f"FALLBACK LOADS SUCCESS: {queries}")
            except Exception as e2:
                print(f"FALLBACK FAILED: {e2}")

    all_queries = [query] + queries
    seen = set()
    final = []
    for q in all_queries:
        q_clean = q.lower()
        if q_clean not in seen:
            seen.add(q_clean)
            final.append(q)
    print(f"FINAL: {final[:num_queries]}")

except Exception as e:
    print(e)

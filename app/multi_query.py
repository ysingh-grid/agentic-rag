import requests
import json
from app.config import settings

def generate_queries(query, num_queries=3):
    url = f"{settings.LLM_BASE_URL}/chat/completions"

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
            "model": settings.LLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You generate search queries."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 200
        }
    )

    try:
        data = response.json()
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        # Fallback if API fails or returns unexpected format
        return [query]

    # --- PARSE ---
    queries = []
    
    # Reasoning models often include <think> blocks in the content.
    # We strip those out before attempting to parse JSON.
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # Or sometimes they use <reasoning>
    elif "</reasoning>" in text:
        text = text.split("</reasoning>")[-1].strip()
        
    try:
        # attempt proper JSON parse
        parsed = json.loads(text)

        if isinstance(parsed, list):
            queries = [q.strip() for q in parsed if isinstance(q, str) and len(q) > 5]

    except Exception:
        # fallback: extract JSON manually if model adds extra text
        start = text.find("[")
        end = text.rfind("]")

        if start != -1 and end != -1:
            try:
                parsed = json.loads(text[start:end+1])
                if isinstance(parsed, list):
                    queries = [q.strip() for q in parsed if isinstance(q, str) and len(q) > 5]
            except Exception:
                queries = []

    # include original query
    all_queries = [query] + queries

    # --- DEDUP ---
    seen = set()
    final = []

    for q in all_queries:
        q_clean = q.lower()
        if q_clean not in seen:
            seen.add(q_clean)
            final.append(q)
    # fallback
    if not final:
        return [query]

    return final[:num_queries]
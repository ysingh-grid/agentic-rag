import httpx

LLM_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "llama-3.2-3b-instruct"


async def call_llm(messages, temperature=0.2, max_tokens=300):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            LLM_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
    data = response.json()
    return data["choices"][0]["message"]["content"]


async def evaluate_answer(query, answer, context):
    prompt = f"""
You are evaluating an answer.

Question:
{query}

Answer:
{answer}

Context:
{context}

Is the answer grounded in the context and correct?

Reply ONLY with:
GOOD
or
BAD
"""

    result = await call_llm([
        {"role": "system", "content": "You are a strict evaluator."},
        {"role": "user", "content": prompt}
    ])

    return "GOOD" in result.upper()


async def rewrite_query(original_query, last_answer):
    prompt = f"""
Rewrite the query to improve retrieval.

Original query:
{original_query}

Previous answer (may be wrong):
{last_answer}

Return ONLY the improved query.
"""

    result = await call_llm([
        {"role": "system", "content": "You improve search queries."},
        {"role": "user", "content": prompt}
    ])

    return result.strip()

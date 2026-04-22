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

Is the answer grounded in the context AND does it make logical sense?
Pay STRICT attention to logical and temporal constraints:
- Sequences or date ranges MUST NOT go backwards in time.
- Quantities, durations, and metrics must be realistic and logically consistent with the context.
- The answer must accurately reflect the specific timeframe, subject, or constraints the user asked about.

If the answer contains absurd logic, self-contradictions, hallucinated constraints, or improperly mixes unassociated facts from the context, you MUST reject it.

Reply ONLY with:
GOOD
or
BAD
"""

    result = await call_llm([
        {"role": "system", "content": "You are a strict evaluator."},
        {"role": "user", "content": prompt}
    ])

    result_upper = result.upper()
    if "BAD" in result_upper:
        return False
    return "GOOD" in result_upper


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

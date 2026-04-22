import logging
import tiktoken

from app.config import settings
from app.session import SessionData

logger = logging.getLogger(__name__)

# Llama-3.2-3B has a large context window; we budget 85% of an 8k footprint.
MAX_TOKENS = int(8000 * 0.85)  # 6800 tokens

# Cache the tiktoken encoder object (thread-safe singleton)
_ENC = None

def count_tokens(text: str) -> int:
    global _ENC
    try:
        if _ENC is None:
            _ENC = tiktoken.get_encoding("cl100k_base")
        return len(_ENC.encode(text))
    except Exception:
        # Fallback: 1 token ≈ 4 characters
        return max(1, len(text) // 4)

def assemble_prompt(
    query: str,
    session: SessionData,
    retrieved_chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Returns (messages_list, trimmed_history_used).

    Budget: system_block + history + query ≤ MAX_TOKENS.
    Trimming removes oldest turns first.
    Empty retrieval: explicit fallback instruction added to system block.
    """
    # --- System block ---
    system_block = (
        "You are a helpful and intelligent AI assistant. "
        "Use the provided document context below to answer the user's question accurately. "
        "If the context does not contain the answer, say so clearly — do not fabricate.\n\n"
        "=== Document Context ===\n"
    )

    if not retrieved_chunks:
        system_block += (
            "No relevant documents were found. "
            "Answer from general knowledge and clearly state you are NOT drawing from the knowledge base.\n"
        )
    else:
        for i, chunk in enumerate(retrieved_chunks):
            content = chunk.get("content", "")
            source = chunk.get("metadata", {}).get("source", "Unknown")
            system_block += f"[Doc {i+1} — Source: {source}]:\n{content}\n\n"

    # --- Query element ---
    query_content = query

    # --- History block: sliding window, trim oldest to fit budget ---
    # Start from the last 5 turns (10 messages) and trim down if needed
    chat_history = (
        session.chat_history[-10:]
        if len(session.chat_history) > 10
        else list(session.chat_history)
    )

    fixed_tokens = count_tokens(system_block) + count_tokens(query_content)
    if fixed_tokens > MAX_TOKENS:
        logger.warning(
            "System block alone (%d tokens) exceeds budget (%d). Context will be truncated.",
            fixed_tokens, MAX_TOKENS
        )

    while chat_history:
        history_block = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history
        )
        if fixed_tokens + count_tokens(history_block) <= MAX_TOKENS:
            break
        chat_history.pop(0)  # drop oldest message first
        if not chat_history:
            logger.warning("Entire chat history trimmed to fit token budget.")

    # --- Assemble final structured messages ---
    messages = [{"role": "system", "content": system_block}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query_content})

    return messages, chat_history

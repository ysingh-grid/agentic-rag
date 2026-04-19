import os
import json
import asyncio
import time
from typing import List, Dict

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from datasets import Dataset
except ImportError:
    print("ragas or datasets not installed. Please install the `eval` extra dependencies.")
    exit(1)

from app.config import settings
from app.retrieval import retriever, hybrid_retrieve, rerank_candidates, _vector_search

def load_eval_dataset(path: str = "eval_dataset.json") -> List[Dict]:
    if not os.path.exists(path):
        # Create dummy for demonstration
        dummy = [
            {
                "query": "What is the capital of France?",
                "ground_truth_context": ["Paris is the capital and most populous city of France."],
                "ground_truth_answer": "The capital of France is Paris."
            }
        ]
        with open(path, 'w') as f:
            json.dump(dummy, f)
        return dummy
    
    with open(path, 'r') as f:
        return json.load(f)

async def _run_eval(mode: str, queries: List[Dict]):
    """
    Evaluates one of three configurations:
    - vector_only
    - hybrid (RRF)
    - hybrid_reranked
    """
    
    print(f"Running evaluation for mode: {mode}")
    retriever.load_models()
    
    data_samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for q in queries:
        query_text = q['query']
        query_emb = await asyncio.to_thread(retriever.embedding_model.get_text_embedding, query_text)
        
        # Retrieval Phase
        if mode == "vector_only":
            docs = await _vector_search(query_text, query_emb, settings.RETRIEVAL_TOP_K)
            docs = [d[0] for d in docs]
        elif mode == "hybrid":
            docs = await hybrid_retrieve(query_text, query_emb, settings.RETRIEVAL_TOP_K)
        elif mode == "hybrid_reranked":
            docs = await hybrid_retrieve(query_text, query_emb, settings.RETRIEVAL_TOP_K)
            docs = await rerank_candidates(query_text, docs, settings.RERANK_TOP_N)
        else:
            raise ValueError("Unknown mode")
            
        contexts = [doc.get("content", "") for doc in docs]
        
        # In a real eval script, you would query your local LLM or a cloud LLM to get 'answer'
        # For offline RAGAS evaluation against the retrieved context, we need the generator's response.
        # We will mock the generator here to just output something so RAGAS can grade the context
        answer = "MOCK_ANSWER" 
            
        data_samples["question"].append(query_text)
        data_samples["answer"].append(answer)
        data_samples["contexts"].append(contexts)
        data_samples["ground_truth"].append(q['ground_truth_answer'])
        
    dataset = Dataset.from_dict(data_samples)
    
    # Needs OPENAI_API_KEY for the judge LLM (Ragas uses GPT by default to evaluate)
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY is not set. Ragas uses OpenAI by default as a judge.")
        
    try:
        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            raise_exceptions=False
        )
        print(result)
        
        # Save results
        out_path = f"eval_results_{mode}_{int(time.time())}.json"
        with open(out_path, 'w') as f:
            json.dump({k: v for k,v in result.items()}, f, indent=2)
            
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vector_only", "hybrid", "hybrid_reranked"], default="hybrid_reranked")
    args = parser.parse_args()
    
    queries = load_eval_dataset()
    asyncio.run(_run_eval(args.mode, queries))

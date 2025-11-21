from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import wikipedia

# Load models
retriever_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print(f"✅ Retriever model loaded on CPU")

pipeline_device = 0 if torch.cuda.is_available() else -1
generator_model = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=pipeline_device
)
print(f"✅ Generator model loaded. Device: {'GPU' if pipeline_device==0 else 'CPU'}")

# Wikipedia fetch + chunking
def split_into_chunks(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def fetch_wikipedia(query, max_articles=5, chunk_size=150, overlap=30):
    """Fetch Wikipedia paragraphs and split into chunks with overlap"""
    try:
        search_results = wikipedia.search(query, results=max_articles)
        chunks = []
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                paragraphs = [p for p in page.content.split("\n") if len(p) > 20]
                for p in paragraphs:
                    chunks.extend(split_into_chunks(p, chunk_size, overlap))
            except wikipedia.DisambiguationError:
                continue
        return chunks
    except Exception as e:
        print("Error fetching Wikipedia:", e)
        return []

# Retriever with Top-k
def retrieve_topk_contexts(query, knowledge_base, top_k=5):
    """Return top-k chunks based on cosine similarity"""
    knowledge_embeddings = retriever_model.encode(knowledge_base, convert_to_tensor=True)
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, knowledge_embeddings)[0]
    
    top_results = torch.topk(cos_scores, k=min(top_k, len(knowledge_base)))
    top_indices = top_results.indices.tolist()
    
    top_chunks = [knowledge_base[i] for i in top_indices]
    return top_chunks

# Generator
def generate_answer(question, context):
    result = generator_model(question=question, context=context)
    return result['answer'], result.get('score', 0)

# Full RAG Pipeline
def ask_rag_pipeline(query, top_k=5):
    knowledge_base = fetch_wikipedia(query)
    if not knowledge_base:
        return "No context found.", "", 0
    
    top_chunks = retrieve_topk_contexts(query, knowledge_base, top_k)
    context_for_model = " ".join(top_chunks)
    
    final_answer, confidence = generate_answer(question=query, context=context_for_model)
    return final_answer, context_for_model, confidence

# Example usage
query = "Who came up with the concept of gravity?"

final_answer, retrieved_context, confidence = ask_rag_pipeline(query)

print(f"Query: {query}")
print(f"Retrieved Context (Top chunks):\n{retrieved_context[:500]}...") 
print(f"Answer: {final_answer}")
print(f"Confidence: {confidence}")

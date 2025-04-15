import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

QUERIES_PATH = "../data/manual_test_queries.txt"
DOCS_PATH = "../data/all_data/train_data.jsonl"

def load_queries(path):
    with open(path, 'r') as f:
        lines = f.read().split("\n")

    queries = []
    relevant_docs = []

    for i in range(0, len(lines), 4):
        if i + 2 < len(lines):
            query = lines[i+1].strip("- ")
            doc_id = lines[i+2].split(':')[-1].strip().strip('"')
            queries.append(query)
            relevant_docs.append(str(doc_id))

    return queries, relevant_docs

def load_documents(path):
    documents = []
    ids = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                doc_id = str(item.get("id", ""))
                text = item.get("description", "") + " " + item.get("title", "") + " " + item.get("feature_summary", "")
                ids.append(doc_id)
                documents.append(text)
            except json.JSONDecodeError:
                continue
    return ids, documents

def precision_at_k(relevant, retrieved, k):
    return sum([1 for doc in retrieved[:k] if doc in relevant]) / k

def average_precision(relevant, retrieved):
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / hits if hits > 0 else 0

def reciprocal_rank(relevant, retrieved):
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1 / (i + 1)
    return 0

def main():
    queries, ground_truth = load_queries(QUERIES_PATH)
    doc_ids, docs = load_documents(DOCS_PATH)

    vectorizer = TfidfVectorizer(stop_words='english')
    doc_vectors = vectorizer.fit_transform(docs)

    results = {
        'P@1': [], 'P@3': [], 'P@5': [],
        'MAP': [], 'MRR': []
    }

    for query, rel_id in zip(queries, ground_truth):
        if rel_id not in doc_ids:
            print(f"  Skipping query: '{query}' — relevant doc ID {rel_id} not found in document collection.")
            continue

        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, doc_vectors).flatten()
        ranked_indices = sims.argsort()[::-1]
        ranked_ids = [doc_ids[i] for i in ranked_indices]

        relevant_set = {rel_id}

        print(f"\nQuery: {query}")
        print(f"Relevant: {rel_id}")
        print(f"Rank: {ranked_ids.index(rel_id) + 1 if rel_id in ranked_ids else 'N/A'}")

        results['P@1'].append(precision_at_k(relevant_set, ranked_ids, 1))
        results['P@3'].append(precision_at_k(relevant_set, ranked_ids, 3))
        results['P@5'].append(precision_at_k(relevant_set, ranked_ids, 5))
        results['MAP'].append(average_precision(relevant_set, ranked_ids))
        results['MRR'].append(reciprocal_rank(relevant_set, ranked_ids))

    print("\n--- Evaluation Results ---")
    for metric in results:
        print(f"{metric}: {np.mean(results[metric]):.4f}")

if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import networkx as nx
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_questions(csv_file):

    df = pd.read_csv(csv_file)

    if "question" not in df.columns:
        raise ValueError("CSV must contain a 'question' column")

    return df["question"].dropna().tolist()


def generate_embeddings(questions, model):

    embeddings = model.encode(questions)

    return embeddings


def build_similarity_graph(questions, embeddings, threshold):

    graph = nx.Graph()

    # Add all questions as nodes
    for i in range(len(questions)):
        graph.add_node(i)

    # Compare every pair
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):

            similarity = cosine_similarity(
                [embeddings[i]],
                [embeddings[j]]
            )[0][0]

            # Add edge if above threshold
            if similarity >= threshold:
                graph.add_edge(i, j)

    return graph


def get_clusters(graph):

    clusters = list(nx.connected_components(graph))

    return clusters


def print_clusters(clusters, questions):

    cluster_id = 1

    for cluster in clusters:

        # Ignore single-question clusters
        if len(cluster) < 2:
            continue

        print(f"\nCluster {cluster_id}:")

        for index in cluster:
            print(f"- {questions[index]}")

        cluster_id += 1


def main():

    parser = argparse.ArgumentParser(
        description="Near-duplicate question clustering"
    )

    parser.add_argument(
        "file",
        help="CSV file containing questions"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold"
    )

    args = parser.parse_args()

    print("\nLoading questions...")
    questions = load_questions(args.file)

    print(f"Loaded {len(questions)} questions")

    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings...")
    embeddings = generate_embeddings(questions, model)
    # Ensure embeddings are a numpy array for downstream operations
    embeddings = np.array(embeddings)

    # Diagnostic: compute basic similarity stats to help choose a threshold
    pairwise = cosine_similarity(embeddings)
    # ignore self-similarity on diagonal when computing max
    n = pairwise.shape[0]
    if n > 1:
        max_sim = np.max(pairwise - np.eye(n))
        print(f"\nEmbedding similarity stats: max_nonself={max_sim:.4f}")
        if max_sim < args.threshold:
            print(f"Note: current threshold={args.threshold} is above the observed max similarity.")
            print(f"Try a lower threshold (e.g. {max_sim - 0.05:.2f} to {max_sim:.2f}).")

    print("Building similarity graph...")
    graph = build_similarity_graph(
        questions,
        embeddings,
        args.threshold
    )

    print("Finding clusters...")
    clusters = get_clusters(graph)

    print("\nDuplicate Clusters:")
    print_clusters(clusters, questions)


if __name__ == "__main__":
    main()
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Model loaded successfully!\n")


# Sample questions
questions = [
    "How to lose weight?",
    "How can I reduce body fat?",
    "Best laptop for coding?",
    "Why is the sky blue?"
]


# Generate embeddings
embeddings = model.encode(questions)

print("Embeddings generated!\n")


# Compare similarities
for i in range(len(questions)):
    for j in range(i + 1, len(questions)):

        similarity = cosine_similarity(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        print(f"Q1: {questions[i]}")
        print(f"Q2: {questions[j]}")
        print(f"Similarity: {similarity:.4f}")
        print("-" * 50)


        

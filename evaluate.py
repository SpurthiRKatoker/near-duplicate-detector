from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from tqdm import tqdm


# Thresholds to test
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]

# Split sizes
TUNE_SIZE = 700
TEST_SIZE = 300


def load_qqp_sample(tune_size=700, test_size=300):

    print("Loading QQP dataset...")

    dataset = load_dataset("AlekseyKorshuk/quora-question-pairs")
    train_split = dataset["train"]

    total = tune_size + test_size

    train_split = train_split.shuffle(seed=42)
    sample = train_split.select(range(total))

    tune_set = sample.select(range(tune_size))
    test_set = sample.select(range(tune_size, total))

    print(f"  Tuning slice:  {len(tune_set)} pairs")
    print(f"  Test slice:    {len(test_set)} pairs (held out)")

    return tune_set, test_set


def compute_predictions(dataset, model, threshold):

    predictions = []
    labels = []

    # Collect all unique questions
    all_questions = set()
    for item in dataset:
        all_questions.add(item["question1"])
        all_questions.add(item["question2"])

    all_questions = list(all_questions)

    print(f"\n  Embedding {len(all_questions)} unique questions...")

    # Generate embeddings ONCE
    embeddings = model.encode(
        all_questions,
        show_progress_bar=True,
        batch_size=64
    )

    # Store in lookup dict
    embedding_map = {
        question: embedding
        for question, embedding in zip(all_questions, embeddings)
    }

    # Score each pair
    for item in tqdm(dataset, desc="  Scoring pairs"):

        q1 = item["question1"]
        q2 = item["question2"]
        label = item["is_duplicate"]  # 1 = duplicate, 0 = not

        similarity = cosine_similarity(
            [embedding_map[q1]],
            [embedding_map[q2]]
        )[0][0]

        prediction = 1 if similarity >= threshold else 0

        predictions.append(prediction)
        labels.append(label)

    return predictions, labels


def evaluate_thresholds(dataset, model, label=""):

    print(f"\n{'='*52}")
    print(f"Evaluation Results — {label}")
    print(f"{'='*52}")

    print(
        f"\n{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1 Score':<12}"
    )
    print("-" * 48)

    best_f1 = -1
    best_threshold = None
    results = []

    for threshold in THRESHOLDS:

        print(f"\nThreshold {threshold}:")
        predictions, labels = compute_predictions(dataset, model, threshold)

        precision = precision_score(labels, predictions, zero_division=0)
        recall    = recall_score(labels, predictions, zero_division=0)
        f1        = f1_score(labels, predictions, zero_division=0)

        results.append((threshold, precision, recall, f1))

        print(
            f"  {threshold:<10}"
            f"  {precision:<10.4f}"
            f"  {recall:<10.4f}"
            f"  {f1:<10.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n→ Best F1 on tuning slice: {best_f1:.4f} at threshold {best_threshold}")
    return results, best_threshold


def main():

    tune_set, test_set = load_qqp_sample(
        tune_size=TUNE_SIZE,
        test_size=TEST_SIZE
    )

    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    start_time = time.time()

    # Step 1: sweep thresholds on tuning slice
    tune_results, recommended_threshold = evaluate_thresholds(
        tune_set, model, label="Tuning Slice (700 pairs)"
    )

    # Step 2: final evaluation on held-out test slice
    print(f"\n{'='*52}")
    print(f"Final Evaluation — Held-out Test Slice (300 pairs)")
    print(f"Using recommended threshold: {recommended_threshold}")
    print(f"{'='*52}")

    predictions, labels = compute_predictions(test_set, model, recommended_threshold)

    precision = precision_score(labels, predictions, zero_division=0)
    recall    = recall_score(labels, predictions, zero_division=0)
    f1        = f1_score(labels, predictions, zero_division=0)

    print(f"\n  Threshold:  {recommended_threshold}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1 Score:   {f1:.4f}")

    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

    # Markdown table for RESULTS.md
    print(f"\n\n--- Paste this into RESULTS.md ---\n")
    print("| Threshold | Precision | Recall | F1 Score |")
    print("|-----------|-----------|--------|----------|")
    for t, p, r, f in tune_results:
        print(f"| {t}       | {p:.4f}    | {r:.4f} | {f:.4f}   |")
    print(f"\n**Recommended threshold:** {recommended_threshold} (best F1 on tuning slice)")
    print(f"**Held-out test F1:** {f1:.4f}")


if __name__ == "__main__":
    main()
# Near-Duplicate Question Detector

A command-line tool that groups near-duplicate questions into clusters using sentence embeddings and cosine similarity. Built and evaluated against the Quora Question Pairs (QQP) dataset.

---

## How It Works

1. Each question is embedded using `all-MiniLM-L6-v2` (a lightweight sentence transformer)
2. Cosine similarity is computed between every pair of questions
3. Any pair above the similarity threshold gets an edge in a graph
4. Connected components of that graph become clusters — questions in the same cluster are considered near-duplicates

---

## Setup

**Requirements:** Python 3.10+

STEP 1:
```bash
python -m venv venv                                              # Create and activate a virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

STEP 2:
pip install sentence-transformers scikit-learn networkx pandas datasets        # Install dependencies
```
---
STEP 3:
## Running the Tool

### Cluster a CSV of questions

```bash
python dedupe.py questions.csv --threshold 0.85
```

- `questions.csv` must have a column named `question`
- `--threshold` is optional, defaults to `0.85`
- Only clusters with 2+ questions are printed




Example output:
```
Loading questions...
Loaded 9 questions

Loading embedding model...
Generating embeddings...
Building similarity graph...
Finding clusters...

Duplicate Clusters:

Cluster 1:
- How to lose weight?
- How can I reduce body fat?

Cluster 2:
- Best laptop for coding?
- Which coding laptop is best?

Cluster 3:
- Why is the sky blue?
- Why does the sky appear blue?
```


STEP 4:
### Run the evaluation against QQP

```bash
python evaluate.py
```

This will: Load the QQP , train test spliting of the dataset, final evaluation , evaluvates the performance on different threshold and print the best threshold 

---

## Project Structure

```
project/
├── dedupe.py              # Main clustering tool
├── evaluate.py            # QQP evaluation harness
├── text_embeddings.py     # Sanity-check script (Day 1)
├── sample_question.csv    # Small demo question set
├── RESULTS.md             # Metrics table and threshold recommendation
├── README.md              # This file
└── requirements.txt       # Dependencies
```

---


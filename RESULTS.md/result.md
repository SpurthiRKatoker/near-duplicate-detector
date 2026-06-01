
# Results

Evaluation against the Quora Question Pairs (QQP) dataset.  
**Setup:** 1,000 pairs sampled, shuffled (seed=42). 700 used for threshold tuning, 300 held out for final evaluation.  
**Model:** `all-MiniLM-L6-v2` (pretrained, no fine-tuning)

---

## Metrics Table (Tuning Slice — 700 pairs)

| Threshold | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| 0.70      | 0.5660    | 0.9486 | 0.7090   |
| 0.75      | 0.6022    | 0.8854 | **0.7168** |
| 0.80      | 0.6545    | 0.7787 | 0.7112   |
| 0.85      | 0.7064    | 0.6561 | 0.6803   |
| 0.90      | 0.8151    | 0.4704 | 0.5965   |

**Recommended threshold:** `0.75`  
**Held-out test F1:** `0.7388`

---

## Tradeoff Analysis

**Low threshold (0.70):** Recall is very high (0.9486) — the tool catches almost every real duplicate. But precision collapses to 0.5660, meaning nearly half of all predicted duplicates are wrong. Good if missing a duplicate is costly; bad if wrongly merging distinct questions causes problems.

**High threshold (0.90):** Precision jumps to 0.8151 — when the tool says "duplicate," it's usually right. But recall drops to 0.4704, meaning it misses more than half of all actual duplicates. Good for a conservative merge workflow; bad if coverage matters.

**The curve is asymmetric:** Precision improves steadily as threshold rises, but recall falls off sharply above 0.80. This means tightening the threshold past 0.80 costs a lot of recall for modest precision gains — the 0.85 and 0.90 rows are a bad trade for most use cases.

---

## Recommended Threshold: 0.75

`0.75` gives the best F1 on the tuning slice (0.7168) and holds up on the held-out test set (0.7388) — actually improving slightly, which suggests the tuning slice was representative and the threshold isn't overfit.

**Why 0.75 over 0.80?**  
The F1 difference is small (0.7168 vs 0.7112), but 0.75 retains meaningfully higher recall (0.8854 vs 0.7787) while sacrificing only a little precision. For a content deduplication use case — where missing a duplicate means the same question gets answered twice — that recall advantage is worth it. A human reviewer can catch the occasional false merge; they can't easily catch duplicates the tool never flagged.

**When to use a different threshold:**
- Use `0.85` if you want high-confidence merges only and are comfortable missing duplicates
- Use `0.70` if recall is paramount and you have a human review step to filter false positives
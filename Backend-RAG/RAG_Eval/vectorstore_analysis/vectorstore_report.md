# Vector Store Parameter Experiment Report

**Experiment Type:** full
**Timestamp:** 2025-12-15T17:32:44.659173
**Documents:** 10
**Test Cases:** 30

---

## Summary

**Best by Precision@K:** chunk_1000_baseline (0.400)
**Best by Hit Rate:** chunk_1000_baseline (0.700)

---

## Detailed Results

| Configuration | Chunk Size | Overlap | Chunks | Precision@K | Hit Rate | MRR | Build Time (s) |
|--------------|------------|---------|--------|-------------|----------|-----|----------------|
| chunk_500 | 500 | 100 | 25 | 0.367 | 0.500 | 0.478 | 0.4 |
| chunk_1000_baseline | 1000 | 200 | 10 | 0.400 | 0.700 | 0.567 | 0.1 |
| chunk_1500 | 1500 | 300 | 10 | 0.400 | 0.700 | 0.567 | 0.1 |
| chunk_2000 | 2000 | 400 | 10 | 0.400 | 0.700 | 0.567 | 0.1 |
| overlap_0 | 1000 | 0 | 10 | 0.400 | 0.700 | 0.567 | 0.1 |
| overlap_100 | 1000 | 100 | 10 | 0.400 | 0.700 | 0.567 | 0.1 |
| overlap_200_baseline | 1000 | 200 | 10 | 0.400 | 0.700 | 0.567 | 0.1 |
| overlap_300 | 1000 | 300 | 10 | 0.400 | 0.700 | 0.567 | 0.1 |
| recursive_default | 1000 | 200 | 10 | 0.400 | 0.700 | 0.567 | 0.1 |
| recursive_sentence | 1000 | 200 | 10 | 0.400 | 0.700 | 0.567 | 0.1 |
| token_based | 256 | 50 | 16 | 0.383 | 0.600 | 0.517 | 0.3 |
| recursive_paragraph | 1000 | 200 | 10 | 0.400 | 0.700 | 0.567 | 0.1 |

---

## Chunk Statistics

| Configuration | Total Chunks | Avg Size | Min Size | Max Size | Std Dev |
|--------------|--------------|----------|----------|----------|---------|
| chunk_500 | 25 | 322 | 48 | 489 | 157 |
| chunk_1000_baseline | 10 | 741 | 666 | 801 | 40 |
| chunk_1500 | 10 | 741 | 666 | 801 | 40 |
| chunk_2000 | 10 | 741 | 666 | 801 | 40 |
| overlap_0 | 10 | 741 | 666 | 801 | 40 |
| overlap_100 | 10 | 741 | 666 | 801 | 40 |
| overlap_200_baseline | 10 | 741 | 666 | 801 | 40 |
| overlap_300 | 10 | 741 | 666 | 801 | 40 |
| recursive_default | 10 | 741 | 666 | 801 | 40 |
| recursive_sentence | 10 | 741 | 666 | 801 | 40 |
| token_based | 16 | 510 | 173 | 801 | 216 |
| recursive_paragraph | 10 | 741 | 666 | 801 | 40 |

---

## Recommendations

Based on the experiment results, **chunk_1000_baseline** provides the best retrieval precision.
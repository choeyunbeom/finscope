# Critic Agent Evaluation Report

**Can LLM-as-Judge Actually Catch Hallucinations?**

Date: 2026-03-17 | Models: Groq llama-3.3-70b-versatile / llama-3.1-8b-instant

---

## 1. Objective

finscope's Critic Agent detects hallucinations in Analyzer-generated financial analyses.
If more than 30% of claims are unsupported by source documents, it returns `insufficient` and triggers a retry.

**Research question: Does this mechanism actually work?**

- Sensitivity: How well does it catch fabricated analyses?
- Specificity: How well does it pass through clean analyses?
- Does model size affect Judge performance?

---

## 2. Experimental Design

### 2.1 Data

8 real AAPL 10-K chunks from ChromaDB used as source excerpts.

### 2.2 Synthetic Test Cases (9 per model run)

| Type | Count | Description | Expected Verdict |
|---|---|---|---|
| **Clean** | 3 | Analysis citing only facts from source excerpts | sufficient |
| **Hallucinated** | 3 | 4 fabricated claims injected (non-existent numbers/facts) | insufficient |
| **Borderline** | 3 | 10 claims, 3 with numbers altered by 5-15% (~30% threshold) | borderline |

### 2.3 Evaluation Setup

- Same Critic prompt used across all cases
- Two models tested: `llama-3.3-70b-versatile` (70B) and `llama-3.1-8b-instant` (8B)
- temperature=0.3, 9 cases per model, 18 total evaluations
- All runs traced via Langfuse (`critic-eval-experiment`)

---

## 3. Results

### 3.1 Confusion Matrix

**70B (llama-3.3-70b-versatile)**

|  | Predicted sufficient | Predicted insufficient |
|---|---|---|
| **Actual clean** | 2 (TN) | 1 (FP) |
| **Actual hallucinated** | 0 (FN) | 3 (TP) |

**8B (llama-3.1-8b-instant)**

|  | Predicted sufficient | Predicted insufficient |
|---|---|---|
| **Actual clean** | 2 (TN) | 1 (FP) |
| **Actual hallucinated** | 1 (FN) | 2 (TP) |

### 3.2 Summary Metrics

| Metric | 70B | 8B |
|---|---|---|
| **Sensitivity** (hallucination detection) | **100%** (3/3) | 67% (2/3) |
| **Specificity** (clean pass-through) | 67% (2/3) | 67% (2/3) |
| **Accuracy** | **83%** | 67% |
| **Borderline detection** | 3/3 insufficient | 0/3 (all sufficient) |

### 3.3 Per-Case Detail (70B)

| Case | Type | Verdict | Cited | Uncited | Correct |
|---|---|---|---|---|---|
| clean_1 | clean | sufficient | 7 | 3 | Y |
| clean_2 | clean | sufficient | 8 | 2 | Y |
| clean_3 | clean | **insufficient** | 7 | 4 | **N (FP)** |
| hallucinated_1 | hallucinated | insufficient | 0 | 9 | Y |
| hallucinated_2 | hallucinated | insufficient | 2 | 5 | Y |
| hallucinated_3 | hallucinated | insufficient | 0 | 9 | Y |
| borderline_1 | borderline | insufficient | 6 | 4 | - |
| borderline_2 | borderline | insufficient | 7 | 3 | - |
| borderline_3 | borderline | insufficient | 7 | 3 | - |

### 3.4 Per-Case Detail (8B)

| Case | Type | Verdict | Cited | Uncited | Correct |
|---|---|---|---|---|---|
| clean_1 | clean | **insufficient** | 2 | 2 | **N (FP)** |
| clean_2 | clean | sufficient | 4 | 0 | Y |
| clean_3 | clean | sufficient | 6 | 0 | Y |
| hallucinated_1 | hallucinated | insufficient | 0 | 5 | Y |
| hallucinated_2 | hallucinated | **sufficient** | 7 | 5 | **N (FN)** |
| hallucinated_3 | hallucinated | insufficient | 4 | 3 | Y |
| borderline_1 | borderline | sufficient | 7 | 3 | - |
| borderline_2 | borderline | sufficient | 7 | 3 | - |
| borderline_3 | borderline | sufficient | 8 | 2 | - |

---

## 4. Analysis

### 4.1 70B is Conservative, 8B is Lenient

- **70B**: Catches all hallucinations (sensitivity 100%), but flags 1 clean analysis as insufficient (clean_3: cited=7, uncited=4 → 4/11=36%, exceeding the 30% threshold). Tends to **reject conservatively** near the boundary.
- **8B**: Lets hallucinated_2 pass as sufficient — cited=7, uncited=5, yet verdict is sufficient. 5/12=42% uncited but still passes. **Fails at ratio calculation.**

### 4.2 Borderline Behaviour is the Key Differentiator

- **70B**: All 3 borderline cases (30% altered numbers) flagged as insufficient — accurately identifies subtle numeric distortions (5-15% changes).
- **8B**: All 3 borderline cases pass as sufficient — completely fails to detect minor numeric alterations.

This is the most significant finding: model size has a dramatic impact on LLM-as-Judge performance at the decision boundary.

### 4.3 False Positive Problem

Both models show specificity of only 67% — each flags 1 clean analysis incorrectly. Root causes:
- When the LLM paraphrases source content, the Critic classifies the paraphrase as "uncited"
- The 30% threshold may be too strict for paraphrased content
- Potential improvements: adjust threshold to 35-40%, or introduce a "partially cited" category

### 4.4 Limitations

- **n=9 per model**: Low statistical significance. Should expand to 30-50+ cases.
- **Same LLM generates and evaluates**: Self-evaluation bias — the generator and judge share similar patterns.
- **temperature=0.3**: Non-deterministic; same case may yield different results on re-run.
- **Synthetic hallucinations**: Fabricated claims may be easier to detect than real-world subtle hallucinations.

---

## 5. Langfuse Tracing

All experiments are traced in Langfuse:
- Trace name: `critic-eval-experiment`
- Observation hierarchy: `critic-eval-experiment` → `critic-eval-judge` → `critic-eval-llm-call`
- Total latency: ~297s (70B), ~280s (8B)
- Per-call latency: 9-24s (70B), 9-22s (8B)
- Trace ID (70B run): `3db88c160c630ed8b6c5eaff258c49f3`

---

## 6. Conclusion

| Question | Answer |
|---|---|
| Does the Critic catch hallucinations? | **70B: Yes** (sensitivity 100%). 8B: Partially (67%). |
| Does it pass clean analyses? | **Partially** — both models show 67% specificity. FP exists. |
| Does model size matter? | **Significantly** — especially for borderline detection (70B: 3/3, 8B: 0/3). |
| Production readiness? | Use 70B as Critic. Consider raising threshold to 35% to reduce FP. |

### Suggested Follow-up Experiments

1. **Threshold sweep**: Generate ROC curve at 20%, 25%, 30%, 35%, 40% thresholds
2. **Cross-model judge**: Generate with 8B, evaluate with 70B (or vice versa) to remove self-eval bias
3. **Larger sample size**: n=30+ per category for statistical significance
4. **Claim-level evaluation**: Measure per-claim cited/uncited accuracy, not just aggregate verdict

---

## Appendix: Reproduction

```bash
# Set GROQ_API_KEY in .env, then:
uv run python -m experiments.critic_eval

# Output files
experiments/critic_eval_cases.json    # Synthetic test cases
experiments/critic_eval_results.json  # Evaluation results + metrics
experiments/REPORT.md                 # This report
```

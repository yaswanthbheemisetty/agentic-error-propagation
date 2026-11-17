# Breaking the Cascade: Error Propagation & Verification Topologies in Multi-Step Agentic LLM Workflows

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14.1](https://img.shields.io/badge/PyTorch-2.14.1%2Bcu130-ee4c2c.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B--Instruct-teal.svg)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Benchmark](https://img.shields.io/badge/Dataset-HumanEval%20(N%3D25)-orange.svg)](https://huggingface.co/datasets/openai/openai_humaneval)

An empirical comparative study investigating **cascading error propagation, position vulnerability, and deterministic verification topologies** across 4-stage agentic LLM workflows.

---

## 1. Research Questions

* **RQ1 (Intrinsic Self-Checking Confirmation Bias):** Does step-level self-checking (`Self_Check`) mitigate cascading errors, or does autoregressive conditioning on corrupted prior context induce confirmation bias and syntactic degradation?
* **RQ2 (Decoupled Verification & Context Rollback):** How effectively does external deterministic verification (`Independent_Verifier` via AST parsing + public docstring execution) paired with clean-context rollback reduce Fault Propagation Rate ($FPR$) and improve Trace Recovery Rate ($TRR$) under early vs. late fault injection?
* **RQ3 (Reliability–Compute Trade-Off):** What is the token and latency cost per successful pipeline completion across verification architectures?

---

## 2. Pipeline Architecture & Compared Topologies

Every benchmark task executes through a 4-stage directed pipeline:
1. **Step 1 — Specification Extraction:** Extracts function inputs, return types, and edge cases.
2. **Step 2 — Algorithmic Planning / Schema Handoff:** Produces a 3-step algorithmic plan (or validated JSON SOP schema).
3. **Step 3 — Code Generation:** Synthesizes executable Python code implementing the Step 2 plan.
4. **Step 4 — Verification & Recovery Gate:** Applies one of three controlled verification policies:
   * **`Naive_Chain` (Baseline):** Passes unverified intermediate plans and code straight through without inspection.
   * **`Self_Check` (Intrinsic Reflection):** Prompts the same LLM to inspect and revise its own Step 2 plan and Step 3 code.
   * **`Independent_Verifier` (Decoupled Oracle + Rollback):** Enforces JSON schema validation at Step 2 and runs a deterministic external oracle (`ast.parse` + public `>>>` docstring assertions) at Step 4. On failure, the corrupted Step 2 context is discarded (*clean-context rollback*) before regenerating code.

---

## 3. Verified Experimental Results (225 Traces)

Evaluated on `openai/openai_humaneval` (`test[:25]`, `SEED = 42`, `do_sample = False`) using `Qwen/Qwen2.5-1.5B-Instruct` (`float16`) on an NVIDIA Tesla T4 GPU:

| Condition | Method | Passed / $N$ | Success $S_4$ (95% CI) | $FPR$ (%) | $TRR$ (%) | Avg Tokens | Avg Latency (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`clean`** | `Naive_Chain` | 2 / 25 | 8.0% [2.2–25.0%] | 92.0% | 0.0% | 334.4 | 12.15 |
| **`clean`** | `Self_Check` | 3 / 25 | 12.0% [4.2–30.0%] | 80.0% | 12.0% | 477.0 | 16.96 |
| **`clean`** | **`Independent_Verifier`** | **18 / 25** | **72.0% [52.4–85.7%]** | **28.0%** | **12.0%** | **327.8** | **11.61** |
| **`inject_early_step2`** | `Naive_Chain` | 4 / 25 | 16.0% [6.4–34.7%] | 84.0% | 16.0% | 340.7 | 12.07 |
| **`inject_early_step2`** | `Self_Check` | 6 / 25 | 24.0% [11.5–43.4%] | 76.0% | 24.0% | 483.4 | 17.44 |
| **`inject_early_step2`** | **`Independent_Verifier`** | **15 / 25** | **60.0% [40.7–76.6%]** | **40.0%** | **60.0%** | **428.7** | **15.68** |
| **`inject_late_step3`** | `Naive_Chain` | 0 / 25 | 0.0% [0.0–13.3%] | 100.0% | 0.0% | 334.4 | 11.91 |
| **`inject_late_step3`** | `Self_Check` | 1 / 25 | 4.0% [0.7–19.5%] | 96.0% | 4.0% | 499.1 | 17.78 |
| **`inject_late_step3`** | **`Independent_Verifier`** | **15 / 25** | **60.0% [40.7–76.6%]** | **40.0%** | **60.0%** | **367.9** | **13.02** |

### Key Empirical Takeaways
* **+64.0 pp Clean Gain at Lower Token Cost:** `Independent_Verifier` boosts clean $S_4$ from `8.0%` to `72.0%` ($p \le 3.1 \times 10^{-5}$) while using fewer average tokens (`327.8`) than both `Naive_Chain` (`334.4`) and `Self_Check` (`477.0`).
* **Confirmation Bias in Self-Checking:** `Self_Check` increases terminal `SyntaxError` failures from `31` (`Naive_Chain`) to `48` across 75 traces because self-reflection conditions directly on corrupted upstream tokens.
* **Zero Runtime/Type/Syntax Leakage:** `Independent_Verifier` eliminates `100%` of `IndentationError`, `NameError`, `TypeError`, and `ValueError` failures across all 75 traces.

---

## 4. Quickstart Reproduction

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

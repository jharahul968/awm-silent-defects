# Mining Silent Data Defects in LLM-Synthesized Agent Environments

**COSC 50303: Data Mining — Course Project, Fall 2026**
Rahul Jha · Department of Computer Science, Texas Christian University

📄 [Project Proposal](Project_Proposal.pdf)

---

## Project Overview

[Agent World Model (AWM)](https://arxiv.org/abs/2602.10090) automatically generates 1,000 executable, database-backed environments for training and evaluating tool-using AI agents. Because these environments are synthesized by LLMs, their initial database states can contain two distinct classes of defects:

- **Tier 1 — Explicit crashes:** INSERT statements that cause SQLite runtime exceptions (foreign-key violations, schema mismatches, etc.). These are caught by the database engine at execution time.
- **Tier 2 — Silent semantic corruptions:** INSERT statements that execute successfully (return code 0) but commit logically broken world state — negative product prices, impossible ages, out-of-bounds ratings, anachronistic dates. These bypass the database engine entirely and are the primary mining target.

**Central research question:**
> Can data-mining methods detect SQL-valid but semantically corrupted records in LLM-synthesized agent environments, including defects that ordinary database execution cannot detect?

---

## Reproduction Findings (Tier 1)

Running the official AWM database-reset procedure (`awm env reset_db`) across all 1,000 environments confirms the dataset quality issues noted in the original repository:

| Metric | Result |
|---|---:|
| Environments | 1,000 |
| INSERT statements | 129,328 |
| Explicit failures | 1,374 (1.06%) |
| Environments with failures | 217 / 1,000 (21.7%) |
| Dominant failure mode | FK violations (76.7%) |
| Malformed sample payload | 1 (`dating_1`) |

Pre-execution ML baselines for Tier-1 failure prediction (PR-AUC, environment-grouped split):

| Model | PR-AUC |
|---|---:|
| Majority baseline | 0.018 |
| Static rules | 0.025 |
| Logistic Regression | 0.061 |
| Random Forest | 0.053 |

---

## Tier-2 Silent Defect Detection (Main Contribution)

Since naturally occurring silent corruptions lack ground-truth labels, controlled SQL-valid corruptions are injected with explicit labels. A 25-environment pilot confirmed that **96.2% of injected corruptions are accepted by SQLite with zero errors**.

Corruption types (multiple severity levels each):
- `negative_money` — negative prices, amounts, costs
- `negative_inventory` — negative stock/quantity counts
- `impossible_age` — biologically implausible ages
- `out_of_bounds_rating` — ratings outside valid scale
- `anomalous_date` — dates in historically implausible years

The full experiment (all 1,000 environments) compares 6–7 detection methods under an environment-grouped train/test split:

| Category | Method |
|---|---|
| Simple baseline | Rule-based detector |
| Statistical baseline | Z-score / robust deviation |
| Unsupervised | Isolation Forest |
| Unsupervised | Local Outlier Factor |
| Supervised | Logistic Regression |
| Supervised | Random Forest |
| Supervised | HistGradientBoosting |

Controlled experiments: (1) severity levels, (2) unseen-corruption-type generalization, (3) feature-group ablation.

---

## Repository Structure

```
analysis/                   # All project scripts
  build_insert_audit.py     # Reproduces AWM reset, builds statement-level audit
  evaluate_insert_predictability.py  # Tier-1 ML baselines
  inject_semantic_corruptions.py     # Tier-2 corruption injection + detection
  run_final_checks.py                # Stability checks, error categorization
  results/                  # Computed summaries (heavy files gitignored)
    audit_summary.json
    environment_audit.json
    feasibility_report.md
    final_checks.json
    predictability_results.json
data/                       # Dataset provenance (actual .jsonl files gitignored)
  README.md                 # HuggingFace dataset reference
awm/                        # AWM library (from upstream, minor bug fix applied)
figures/                    # Plots and diagrams
Project_Proposal.pdf        # Course project proposal
```

---

## Setup

```bash
# Install dependencies
uv sync

# Download the AWM dataset from HuggingFace
huggingface-cli download Snowflake/AgentWorldModel-1K \
    --repo-type dataset --local-dir ./data
```

---

## Reproducing the Main Experiments

```bash
# Step 1: Build the statement-level audit (Tier-1 dataset)
uv run python analysis/build_insert_audit.py \
    --data-dir data \
    --output-dir analysis/results

# Step 2: Tier-1 pre-execution failure prediction baselines
uv run python analysis/evaluate_insert_predictability.py

# Step 3: Stability checks and error categorization
uv run python analysis/run_final_checks.py

# Step 4: Tier-2 silent defect injection and detection (full scale — WIP)
uv run python analysis/inject_semantic_corruptions.py
```

Results are written to `analysis/results/`.

---

## Data Source

- **Dataset:** [Snowflake/AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) (HuggingFace, Apache 2.0)
- **Upstream code:** [Snowflake-Labs/agent-world-model](https://github.com/Snowflake-Labs/agent-world-model)
- **MCP benchmark harness:** [Raibows/mcp-adapted-bench](https://github.com/Raibows/mcp-adapted-bench) (BFCLv3, τ²-bench, MCP-Universe)
- **Paper:** Wang et al., "Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning," ICML 2026. [arXiv:2602.10090](https://arxiv.org/abs/2602.10090)

This project extends the AWM artifact for a data-mining study and is not affiliated with Snowflake or the original authors.

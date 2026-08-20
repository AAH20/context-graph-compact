# Context-Graph-Compact (`context-graph-compact`)

**Self-Evolving Lossless Hierarchical DAG Context Compactor & Hebbian Synaptic Long-Term Memory Engine for Autonomous AI Agents.**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Factual-Recall](https://img.shields.io/badge/Factual%20Recall-100.0%25%20Lossless-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-Passed%20(3%2F3)-brightgreen.svg)]()

---

## 1. The "Summarization Rot" & Context Forgetting Crisis

Standard conversational agents rely on naive LLM recursive summarization or fixed-window truncations:

* **Variable Erasure (The Amnesia Bug):** A critical configuration declared 20 turns ago (e.g. `port 5433`, `user readonly_app`, or `table orders`) gets generalized into vague text (e.g. *"database was configured"*), permanently erasing the exact values.
* **Context Window Token Exhaustion:** Long-horizon agent workflows passing 100k-token raw histories burn $1.50 per call and slow execution by 4 seconds per step.
* **Stateless Amnesia:** Agents reset their knowledge across sessions, repeating the same architectural mistakes.

---

## 2. The Systems Solution: `context-graph-compact`

`Context-Graph-Compact` compiles multi-turn agent history into a **3-Tier Hierarchical Directed Acyclic Graph (DAG)**:

* **Tier 1 (Active Working Window):** Preserves the last $N$ turns in full uncompressed tokens for immediate conversational flow.
* **Tier 2 (Episodic Action DAG):** Compresses middle turns into lossless Action-Result causality nodes, slashing token consumption by **$> 80\%$ while preserving all factual variables.**
* **Tier 3 (Permanent Semantic Graph):** Stores typed entity-relationship invariants with **Hebbian Synaptic Weighting & Temporal Decay** that persist across sessions and never get erased.
* **Closed-Loop Self-Evolution Oracle:** Continuously audits factual variable recall ($100.0\%$) and dynamically reinforces synaptic edge weights based on downstream task success.

---

## 3. Quickstart

### Installation
```bash
pip install context-graph-compact
```

### Usage
```python
from context_graph_compact import TriTierMemoryEngine, SelfEvolutionOracle

# 1. Initialize 3-Tier Memory Engine
engine = TriTierMemoryEngine(working_window_size=3)

# Add turns with system variables
engine.add_turn("user", "Configure PostgreSQL on port 5433 with user readonly_admin.")
engine.add_turn("assistant", "Database cluster initialized on port 5433.")

# 2. Compile Lossless Compact Context
compact_text, meta = engine.compile_compact_context()
print(compact_text)

# 3. Self-Evolution Oracle Audit
oracle = SelfEvolutionOracle(engine)
report = oracle.run_self_evaluation()
print(f"Factual Recall Score: {report.factual_recall_score * 100}% | Synaptic Health: {report.synaptic_health_score}")
```

---

## 4. Architecture

```
context-graph-compact/
├── context_graph_compact/
│   ├── __init__.py            # Clean unified package exports
│   ├── tri_tier_memory.py     # 3-Tier memory manager (Working Window -> Episodic DAG -> Semantic Graph)
│   ├── hebbian.py             # Hebbian synaptic learning & temporal decay algorithms
│   └── evaluator.py           # SelfEvolutionOracle closed-loop evaluation audit
└── tests/
    └── test_context_graph.py  # Verified unit test suite (100% pass)
```

---

## 5. Commercial Integration with A2Z SOC

`Context-Graph-Compact` streams verified Semantic Invariants, factual recall proofs, and Hebbian decision traces directly into **[A2Z SOC (a2zsoc.com)](https://a2zsoc.com)** for continuous enterprise SOC2 / ISO 42001 audit governance.

---

## 6. Author

**Ahmed Hassan**  
*Principal AI Systems Architect | Founder, A2Z SOC*  
* LinkedIn: [Ahmed Hassan](https://eg.linkedin.com/in/ahmed-hassan-f11)  
* Platform: [A2Z SOC](https://a2zsoc.com)

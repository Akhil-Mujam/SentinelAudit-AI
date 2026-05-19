# Graph-Orchestrated Contract Audit Engine

A stateful, graph-orchestrated compliance engine designed to automate the extraction and validation of unstructured legal contracts against internal corporate compliance guidelines. Built with **LangGraph**, **Groq (Llama 3.3 70B)**, and local vector retrieval.

---

## 🎯 Architectural Motivation: Why LangGraph?

Standard linear LLM execution pipelines (like sequential LangChain chains) degrade rapidly when applied to document audit tasks. Linear chains offer no mechanism for error recovery or conditional runtime branching. If an intermediate parsing stage generates malformed JSON or hallucinates a metric, the entire process crashes.

This system maps the compliance workflow as a cyclic state graph using **LangGraph**. This structural approach guarantees:
1. **Deterministic State Transitions:** Explicit code-level routing pathways rule management, decoupling application control flow from unreliable LLM prompt instructions.
2. **Dynamic Self-Correction Loops:** If the Auditor node identifies an incomplete extraction or schema deviation, it routes execution backwards to the Extractor node with explicit feedback payloads for runtime self-correction.
3. **Execution Isolation:** Separating extraction, contextual reference retrieval, and critical risk analysis limits the token footprint per node invocation and eliminates token context degradation.

---

## 🚀 The System Topology & Failure Handling

                +-----------------------+
                |        START          |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
+-----------------> |   Extraction Agent    | <--- Implements Runtime Self-Correction
|                   +-----------+-----------+
|                                |
|                                v
|                   +-----------------------+
|                   |  Context Retrieval    | <--- ChromaDB Payload Vector Lookup
|                   +-----------+-----------+
|                                |
|                                v
|                   +-----------------------+
+------------------ |     Auditor Agent     |
(Audit Discrepancy +-----------+-----------+
or Bad Output)                 |
[Routing Gate Edge]
/       |

(Loops >=3)      /         \ (Compliance Passed)
(Or Violation)  /

v             v
+-------------------+ +-------------------+
|    Human Gate     | |   Approve/End     |
+-------------------+ +-------------------+


### Production Resilience & Failure Mitigation
* **Malformed JSON Recovery:** Node outputs are bound to a strict **Pydantic** model via Groq's native `json_schema` compiler. If parsing constraints fail, Python exceptions are trapped locally at the node border and formatted into error-corrective text passed back to the engine for a dynamic retry.
* **Deterministic Circuit Breakers:** To prevent runaway API token expenditures from infinite multi-agent reasoning iterations, a hard evaluation boundary (`loop_count >= 3`) automatically intercepts execution paths and shifts state payloads to a persistent Human-in-the-Loop escalation node.
* **Infrastructure Strategy:** This repository uses a local in-memory instance of ChromaDB utilizing Hugging Face `all-MiniLM-L6-v2` embeddings solely as a zero-infrastructure test double for rapid prototyping. For enterprise horizontal scale and metadata filtering, the code is structured to drop into a persistent distributed index like Qdrant or pgvector.

---

## 📊 Evaluation Metrics

To graduate this architecture from prototype to production status, system validation is measured against an assertion-driven dataset across the following core performance thresholds:

| Metric Group | Primary Indicator | Target Threshold | Measuring Strategy |
| :--- | :--- | :--- | :--- |
| **Data Extraction** | Schema Validation Success % | > 99.5% | Try/Except execution catch on initial Pydantic bounds. |
| **Precision & Recall**| Missing Clause Rate (False Negatives)| 0.0% | Golden validation dataset cross-checks on mandatory risks. |
| **System Safety** | Hallucination Frequency | 0.0% | LLM-as-a-Judge validation via G-Eval / RAGAS frameworks. |
| **Operational Efficiency**| Token Efficiency Ratio | Scalable | Node tracking via native **LangSmith** telemetry spans. |

---

## 🛠️ Quickstart

### 1. Initialize Environment & Dependencies
```bash
git clone [https://github.com/your-username/SentinelAudit-AI.git]
cd SentinelAudit-AI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

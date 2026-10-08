# Practice Knowledge Base for RAG

A curated, modular multi-format dataset designed specifically for Retrieval-Augmented Generation (RAG) experiments, chunking optimization, and evaluation benchmarks.

## Knowledge Base Files

| File | Format | Domain | Key Concepts / Facts |
| :--- | :--- | :--- | :--- |
| [`refund_policy.md`](./refund_policy.md) | Markdown | E-Commerce | 30-day window, eligible packaging, 5-7 day processing, non-refundable exclusions |
| [`shipping_faq.md`](./shipping_faq.md) | Markdown | E-Commerce | Standard (3-5d, $4.99, free over $50), Express (1-2d, $14.99), Overnight ($29.99, 1pm cutoff), US only |
| [`store_hours.txt`](./store_hours.txt) | Plain Text | Operations | Mon-Fri 9am-6pm EST, Sat 10am-4pm, Sun Closed, support lines, holiday closures |
| [`product_pricing.json`](./product_pricing.json) | JSON | SaaS Platform | Basic ($10/mo), Pro ($25/mo, 300 req/min API), Enterprise ($99/mo, 1200 req/min API), 20% annual discount |
| [`api_rate_limits.md`](./api_rate_limits.md) | Markdown | Technical / API | Quotas per plan, HTTP 429 status, Retry-After header, jittered backoff formula |
| [`authentication_guide.md`](./authentication_guide.md) | Markdown | Security | Bearer tokens, 90-day expiration, 60-day rotation, granular scopes, immediate revocation |
| [`troubleshooting_faq.txt`](./troubleshooting_faq.txt) | Plain Text | Engineering | Error codes (`ERR_AUTH_EXPIRED`, `ERR_RATE_EXCEEDED`, `ERR_PAYLOAD_TOO_LARGE`), webhook retry intervals |
| [`system_incidents.json`](./system_incidents.json) | JSON | Post-Mortems | Structured incident history: incident IDs, downtime minutes, root cause analysis, resolutions |
| [`rag_eval_benchmark.json`](./rag_eval_benchmark.json) | JSON | Evaluation | 10 evaluation test cases: single-fact, threshold, pricing, quotas, multi-hop synthesis, negative/hallucination checks |

## RAG Design Principles Applied
1. **Single-Topic Fact Density:** Avoids long, rambling text; each chunk contains specific rules, numbers, or procedures.
2. **Multi-Format Variety:** Tests loaders across `.md`, `.txt`, and structured `.json`.
3. **Structured vs. Unstructured Testing:** Supports vector similarity searches on unstructured text as well as hybrid or metadata-filtered queries on structured JSON data.
4. **Hallucination Benchmarking:** Includes queries intentionally outside the knowledge base to verify that the RAG chain appropriately responds with "I don't know" when ground truth is absent.

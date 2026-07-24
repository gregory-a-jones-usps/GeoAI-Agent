# Request: Dedicated LLM Endpoint for geo-agent Intent Classification

**Requester:** robert.e.brimhall@usps.gov  
**Date:** 2026-07-24  
**App:** geo-agent-dev  
**Current endpoint:** mas-33c3b825-endpoint (dan.p.houston@usps.gov)  

---

## Problem

The geo-agent app needs to classify user questions and extract structured
parameters (intent, layer, location, filters, etc.) from natural language.

The current `mas-33c3b825-endpoint` is an **agent endpoint** with attached
tools (Genie spaces, enterprise sales supervisor). When called for simple
text classification:

1. It attempts tool calls first (Genie, enterprise_sales) — all fail due to
   permissions or irrelevance
2. Only then falls back to the raw LLM for the actual classification
3. Result: **3–16 second latency** for a task that needs <1 second
4. **~25% failure rate** — some queries get fully intercepted by agent tools
   and never produce the structured JSON we need

## What's Needed

A **lightweight model serving endpoint** configured as a raw LLM (no agent
tools, no function calling). Used exclusively for structured extraction.

### Option A: External Model Endpoint (preferred)

Create a model serving endpoint pointing to Azure OpenAI (already in the
Azure tenant):

```json
{
  "name": "geo-intent-extractor",
  "config": {
    "served_entities": [{
      "external_model": {
        "name": "gpt-4o-mini",
        "provider": "azure_openai",
        "azure_openai_config": {
          "azure_deployment_name": "gpt-4o-mini",
          "azure_resource_name": "<resource>",
          "azure_api_version": "2024-08-01-preview"
        }
      }
    }]
  }
}
```

**Why gpt-4o-mini:** Fast (~300ms), cheap, excellent at structured JSON
extraction. No reasoning overhead needed.

### Option B: Provisioned Throughput (if external blocked)

Provision a small model (e.g., Llama 3.1 8B or Qwen 2.5 7B) with 1-2
throughput units. Only needs to handle ~10-50 RPM for the geo-agent.

### Option C: Modify existing MAS endpoint

If creating a new endpoint isn't feasible, enable `tool_choice: "none"` on
`mas-33c3b825-endpoint` so callers can bypass agent tool execution:

```json
{"input": [...], "tool_choice": "none", "max_tokens": 200}
```

This would skip the Genie/supervisor calls and return the raw LLM response
directly.

## Expected Usage

- **Volume:** 10–50 requests/minute during business hours
- **Payload:** ~500 tokens input (system prompt + question), ~150 tokens output
- **Latency target:** <1 second p95
- **Format:** OpenAI-compatible chat API (`messages` field)

## Integration

Once provisioned, we update `app.yaml`:

```yaml
- name: LLM_ENDPOINT
  value: "geo-intent-extractor"   # new endpoint
- name: LLM_AGENT_ENDPOINT
  value: "mas-33c3b825-endpoint"  # keep for complex queries
```

The app uses the fast endpoint for intent classification + filter extraction,
and falls back to the MAS agent endpoint only for complex Genie-routed queries.

## Impact

| Metric | Current (MAS agent) | With dedicated endpoint |
|--------|--------------------|--------------------------|
| Classification latency | 3–16s | <1s |
| Reliability | ~75% | ~99% |
| User-perceived response time | 5–20s | 2–5s |
| Filter coverage | Hardcoded regex | Any natural language |

## Contact

Happy to walk through the prototype — tested in this workspace on 2026-07-24.
The extraction prompt and response parsing are already integrated into
`agent_router.py` (`_extract_filters_llm` method).

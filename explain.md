# Browser Agent — Full Codebase Explanation

This document covers the entire codebase end-to-end and the event-loop bug that was fixed in the most recent change. It is the canonical reference for new contributors and reviewers.

---

## 1. Runtime confirmation: Amazon Bedrock AgentCore

This project runs on **Amazon Bedrock AgentCore**. AgentCore provides the HTTP server, lifecycle management, and request entrypoint; the Bedrock Runtime API provides the LLM tool-calling. Concrete evidence:

| Where | Code |
|---|---|
| `requirements.txt:7` | `bedrock-agentcore>=1.8.0` |
| `orchestrator.py:10` | `from bedrock_agentcore.runtime import BedrockAgentCoreApp` |
| `orchestrator.py:392` | `app = BedrockAgentCoreApp(lifespan=lifespan)` |
| `orchestrator.py:395` | `@app.entrypoint` decorator on the per-request handler |
| `orchestrator.py:412` | `app.run(host=host, port=port)` starts the AgentCore HTTP server |

The Bedrock model itself is invoked via the **Bedrock Runtime Converse API** (`bedrock-runtime` boto3 client) — see `orchestrator.py:85`, `orchestrator.py:279`, `analysis_agent.py:78`, `writing_agent.py:82`.

So the stack is:
- **Hosting / lifecycle / HTTP**: Bedrock AgentCore (`BedrockAgentCoreApp`)
- **LLM tool-calling loop**: Bedrock Runtime Converse API
- **Browser automation**: Playwright (Chromium, headless)
- **Optional external tools**: Model Context Protocol (MCP) over stdio

---

## 2. The bug we fixed (event-loop mismatch)

### Symptom

Every first request hung indefinitely. The orchestrator logged `BrowserAgent received action: navigate` and then never produced any further output. Even the configured 30-second navigation timeout never fired.

### Root cause

`BedrockAgentCoreApp` runs the **lifespan handler** (startup) on one asyncio event loop, and creates a **new event loop for each request handler invocation**. Playwright async objects (`Browser`, `Page`) are bound to the loop they were created on — the underlying CDP pipe is only serviced when that original loop is running. If you call `page.goto()` from a different loop:

1. The call `await`s a future that is associated with the original loop.
2. The original loop is no longer running, so nobody pumps the CDP messages.
3. `goto()` waits forever.
4. `set_default_navigation_timeout()` also schedules its timer on the original loop, so the timeout never fires either.

We confirmed this empirically by logging `id(asyncio.get_running_loop())` in both `initialize()` and `handle()`:

```
initialize  loop_id=137088768751088
handle      loop_id=137088733398128
```

Two different loops, exactly as predicted.

### Fix

`BrowserAgent` no longer launches Chromium during lifespan. It launches lazily on the first `handle()` call within a request, so Playwright always binds to the request's loop. After the request finishes, the orchestrator entrypoint tears down the browser in a `try/finally` so the next request gets a fresh instance on its own loop.

Concrete code changes:

- **`browser_agent.py`**
  - `initialize()` now only loads YAML config — no Playwright launch.
  - New `_ensure_browser()` method launches Playwright, Chromium, and a Page on first call within the request.
  - `handle()` calls `_ensure_browser()` before every action.
  - `shutdown()` is idempotent — safe to call after every request and again at app shutdown.
- **`orchestrator.py:invoke()`** wraps the call to `orch.handle()` in `try/finally` and calls `await orch.browser_agent.shutdown()` in the `finally` block.

### Cost / trade-off

Every request now pays the Chromium cold-start (~1.3s in WSL2 on this machine). Within a single request, multiple `browser_action` tool calls share the same browser instance (the second-and-later calls reuse the existing Page), so there's no per-action launch overhead.

### Side fix in the same change-set

`config.yaml` had two retired Bedrock model IDs:

- `analysis_agent`: `anthropic.claude-3-haiku-20240307-v1:0` — marked **Legacy** (access blocked after 30 days of inactivity).
- `writing_agent`: `anthropic.claude-3-sonnet-20240229-v1:0` — **end-of-life** (hard-retired).

Both also lacked the `eu.` prefix that **cross-region inference profiles** require in `eu-central-1`. Updated to:

- `analysis_agent`: `eu.anthropic.claude-haiku-4-5-20251001-v1:0`
- `writing_agent`: `eu.anthropic.claude-sonnet-4-6`

The orchestrator's own model (`eu.amazon.nova-lite-v1:0`) was already correctly prefixed and was never affected.

---

## 3. Architecture overview

```
              ┌─────────────────────────────────────┐
              │   BedrockAgentCoreApp (HTTP server) │
              │   POST /invocations                 │
              └──────────────┬──────────────────────┘
                             │
                             ▼
                       invoke(payload)
                             │
                             ▼
              ┌─────────────────────────────────────┐
              │   Orchestrator.handle()             │
              │   - System prompt enforces order    │
              │   - Loop ≤ max_turns                │
              │   - Calls Bedrock Converse          │
              │   - Routes tool_use → child agent   │
              └──┬──────────────┬───────────────┬───┘
                 │              │               │
                 ▼              ▼               ▼
           BrowserAgent   AnalysisAgent    WritingAgent
           (Playwright)   (Bedrock LLM)    (Bedrock LLM)
```

**Pipeline order** is enforced by the orchestrator's system prompt (`orchestrator.py:23`):

```
1. browser_action  (navigate + extract — collect raw web content)
2. analysis_action (structure: Summary / Key Facts / Insights)
3. writing_action  (final polished document)
```

The orchestrator does not hard-enforce the order — the LLM can deviate (and sometimes does, e.g. skip analysis for a trivial input). All three child agents implement the same `BaseAgent` interface (`initialize` / `handle` / `shutdown`).

---

## 4. File-by-file walkthrough

### `agent_base.py` — interface contract

Defines `AgentResponse` (TypedDict with `status: "success" | "error"` and optional `message`) and `BaseAgent` (abstract class with `initialize`, `handle`, `shutdown` methods, all `async`). Every child agent implements this contract so the orchestrator can route to them uniformly.

### `browser_agent.py` — Playwright tool

Concrete `BaseAgent` that exposes Chromium as a tool. After the fix, the lifecycle is per-request, not per-process:

- `initialize()` — loads config only.
- `_ensure_browser()` — lazy-launches Playwright + Chromium + Page on first action of a request. Sets `default_timeout` and `default_navigation_timeout` from `config.browser_timeout` (default 30s).
- `handle({action, ...})` — supports four actions:
  - `navigate` — `page.goto(url, wait_until="domcontentloaded")`, returns `{title, url}`.
  - `click` — `page.click(selector)`.
  - `extract` — `page.locator(selector).inner_text()`, returns `{text}`.
  - `screenshot` — `page.screenshot(path=...)`, returns `{path}`.
- `shutdown()` — closes Page → Browser → Playwright, idempotent (each step wrapped in `with suppress(Exception)`). Called by the orchestrator at the end of every request and again at app shutdown.

### `analysis_agent.py` — Bedrock LLM analyser

Stateless agent that calls a Bedrock Converse model to structure raw text into **Summary / Key Facts / Notable Insights**.

- `initialize()` — loads config and creates a `bedrock-runtime` boto3 client (uses configured `aws_profile`, falls back to default credentials).
- `handle({content, focus?})` — wraps the synchronous `bedrock_client.converse()` call in `asyncio.to_thread()` and `asyncio.wait_for(timeout=bedrock_timeout_s)` so a hung Bedrock call cannot block the event loop. Returns `{status, analysis}` or `{status, message}`.
- `shutdown()` — no-op (no persistent resources).

### `writing_agent.py` — Bedrock LLM writer

Same shape as `AnalysisAgent`. Takes a structured analysis and produces a polished document.

- `handle({analysis, format?, tone?})` — `format` ∈ `{markdown, plain, html}`, `tone` ∈ `{professional, casual, technical}`. Defaults from `config.yaml` (`default_format: markdown`, `default_tone: professional`). Returns `{status, document}`.

### `orchestrator.py` — the engine

The largest file. Lays out the AgentCore app, the Bedrock tool-use loop, and the wiring between everything.

#### Top-level
- Loads `.env` via `python-dotenv`.
- Defines `ORCHESTRATOR_SYSTEM_PROMPT` enforcing the pipeline order.
- Holds a global `orch: Orchestrator` instance set up by the lifespan handler.

#### `Orchestrator.initialize()`
- Reads `config.yaml`.
- Decides stateless mode (`memory.mode`) — when stateless, MCP servers are skipped entirely.
- Builds the `bedrock-runtime` client.
- Calls `initialize()` on each child agent.
- For each MCP server in `config.mcp_servers`, spawns the subprocess via `stdio_client`, opens a `ClientSession`, calls `session.list_tools()`, and translates each tool into Bedrock `toolSpec` format. Stores the `tool_name → server_name` mapping for later dispatch.

#### `Orchestrator._get_tool_config()`
Builds the Bedrock `toolConfig` payload by combining three hand-written tool specs (`browser_action`, `analysis_action`, `writing_action`) with the dynamically-discovered MCP tools.

#### `Orchestrator._scrub_content()`
Optional defence against indirect prompt injection from scraped pages. If `config.guardrail_id` is set, the scraped text from `browser_action` is run through Bedrock Guardrails before being fed back to the model. If the guardrail intervenes the content is replaced with `"[Content blocked by Bedrock Guardrail]"`. No-op when no guardrail is configured.

#### `Orchestrator.handle({task})` — the tool-use loop

This is the ReAct loop. Up to `max_turns` (default 15) iterations of:

1. Call `bedrock_client.converse(modelId, system, messages, toolConfig)` via `asyncio.to_thread` with a timeout (`bedrock_timeout_s`, default 60s).
2. Inspect `response["stopReason"]`:
   - **`tool_use`** — for each `toolUse` block in the message, dispatch:
     - `browser_action` → `self.browser_agent.handle(input)`. On success, scrub `text` and `title` fields with `_scrub_content`.
     - `analysis_action` → `self.analysis_agent.handle(input)` (with empty-content guard).
     - `writing_action` → `self.writing_agent.handle(input)` (with empty-analysis guard).
     - MCP tools → `session.call_tool(tool_name, input)` via the cached `mcp_tool_map`.
   - Pack each result into a `toolResult` message and append to the conversation. Loop again.
   - **anything else** — extract the final `text`, strip optional `<thinking>/<response>` wrappers, return `{status: success, response: text}`.
3. If the loop exhausts `max_turns`, return an error.

#### `Orchestrator.shutdown()`
Closes the `AsyncExitStack` (which tears down all MCP subprocesses) and calls `shutdown()` on each child agent.

#### Lifespan + entrypoint
- `lifespan(app)` — instantiates and initialises the global `Orchestrator` at startup; calls `orch.shutdown()` on app shutdown.
- `invoke(payload)` — `@app.entrypoint`-decorated handler. Reads `payload["prompt"]` or `payload["task"]`, calls `orch.handle({task})`, and **always** calls `orch.browser_agent.shutdown()` in a `finally` so the per-request browser is released. **This `finally` is half of the event-loop fix** — without it the browser would leak across requests on a stale loop.
- `app.run(host, port)` — boots the AgentCore HTTP server. Falls back to `app.run()` for older AgentCore versions that don't accept host/port kwargs.

### `config.yaml` — central configuration

| Key | Purpose | Current value |
|---|---|---|
| `region` | AWS region for Bedrock | `eu-central-1` |
| `model_id` | Orchestrator's LLM (cross-region inference profile) | `eu.amazon.nova-lite-v1:0` |
| `aws_profile` | boto3 profile name | `default` |
| `headless` | Playwright Chromium headless mode | `true` |
| `browser_timeout` | Default page timeout (seconds) | `30` |
| `max_turns` | Cap on Bedrock tool-use turns per request | `15` |
| `bedrock_timeout_s` | Per-`Converse` call timeout | `60` |
| `memory.mode` | `stateless` disables MCP servers | `stateless` |
| `guardrail_id` | Optional Bedrock Guardrail for scraped content | empty (disabled) |
| `analysis_agent.model_id` | Sub-agent model | `eu.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `writing_agent.model_id` | Sub-agent model | `eu.anthropic.claude-sonnet-4-6` |
| `writing_agent.default_format` | Output format | `markdown` |
| `writing_agent.default_tone` | Output tone | `professional` |
| `mcp_servers` | External MCP servers (only used in non-stateless mode) | `sqlite` example |

### `requirements.txt` — runtime dependencies

- `bedrock-agentcore>=1.8.0` — the HTTP server / entrypoint framework.
- `boto3>=1.34.0` — Bedrock Runtime client.
- `playwright>=1.42.0` — browser automation.
- `mcp>=1.1.2` — Model Context Protocol client (for `stdio_client` / `ClientSession`).
- `pyyaml`, `loguru`, `python-dotenv`, `websockets`.

### `aws/` — AgentCore Runtime distribution

Mirror of the AgentCore Runtime install package — used for deploying this app to the managed AgentCore Runtime service, separate from local development.

---

## 5. End-to-end request walk-through

Take the test request used during the bug investigation:

```bash
curl -X POST http://127.0.0.1:8081/invocations \
  -H "Content-Type: application/json" \
  -d '{"task":"Navigate to https://www.wikipedia.org, extract h1, then write a short markdown summary."}'
```

What happens, with timings from the actual run:

1. **Lifespan (once at startup)** — `Orchestrator.initialize()` loads config, builds the Bedrock client, runs each child's `initialize()` (browser config-load only, no Chromium yet), skips MCP because stateless mode.
2. **Request arrives** — AgentCore creates a fresh event loop and calls `invoke(payload)`.
3. **Turn 1** — Orchestrator sends the task + tool specs to `eu.amazon.nova-lite-v1:0`. Model responds with `tool_use: browser_action(navigate, https://www.wikipedia.org)`.
4. **`browser_action navigate`** — `BrowserAgent.handle()` calls `_ensure_browser()` (cold-launches Chromium → 1.3s), then `page.goto(...)` (4.1s for Wikipedia load). Returns `{title: "Wikipedia", url: ...}`. Total **5.4s**.
5. **Tool result fed back** — Orchestrator wraps the result as `toolResult` and re-converses. Model decides next step.
6. **Same turn, second tool call** — `tool_use: browser_action(extract, h1)`. `_ensure_browser()` is a no-op now (Page already exists). `inner_text()` runs in **0.2s**, returns `{text: "Wikipedia\nThe Free Encyclopedia"}`.
7. **Turn 2** — Model receives both tool results. Decides to skip `analysis_action` (small input — LLM judgment) and goes straight to `tool_use: writing_action(format=markdown, analysis="Wikipedia...")`.
8. **`writing_action`** — `WritingAgent` calls `eu.anthropic.claude-sonnet-4-6` via Bedrock Converse (in `to_thread`, with 60s timeout). Returns `{document: "..."}`. **21s** of LLM latency.
9. **Turn 3** — Model has the document and emits a final `text` block with `stopReason != "tool_use"`. Orchestrator returns `{status: "success", response: <markdown>}`.
10. **`finally` block in `invoke`** — calls `orch.browser_agent.shutdown()`. Page → Browser → Playwright torn down. Next request starts cold.

Total round-trip: **39.6s** from `POST` to `200 OK`.

---

## 6. Operational notes

- **Dev loop**: `PORT=8081 python3 orchestrator.py` runs the AgentCore HTTP server locally. `POST /invocations` with `{"task": "..."}` (or `{"prompt": "..."}`).
- **Permissions required**: `bedrock:InvokeModel` and `bedrock:Converse` for the IAM principal in `config.aws_profile`. Each cross-region inference profile in `config.yaml` must be enabled in the Bedrock console under **Model access**.
- **Listing available models**: `aws bedrock list-inference-profiles --region eu-central-1 --query "inferenceProfileSummaries[?contains(inferenceProfileId, 'anthropic')].inferenceProfileId"`.
- **Process hygiene**: a misbehaving Playwright run can leave orphan `chrome-headless-shell` and `playwright/driver/node` processes. Kill with `pkill -9 -f "chrome-headless-shell"; pkill -9 -f "playwright/driver/node"`.
- **Stateless vs stateful**: set `memory.mode: stateful` in `config.yaml` to enable MCP servers (e.g. `sqlite` for persistence). The MCP wiring in `orchestrator.initialize()` will then connect to each configured server and expose its tools to the LLM alongside the local agents.

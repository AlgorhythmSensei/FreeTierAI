# FreeTierAI — LLM Provider Comparison Tool

A Streamlit app for comparing free-tier LLM providers side-by-side. No paid models, no hidden costs.

---

## Providers

| Provider | API Console | Free Model Tier |
|---|---|---|
| **Groq** | https://console.groq.com/keys | Yes — high-speed inference |
| **OpenRouter** | https://openrouter.ai/keys | Yes — `:free` suffix models only, fetched live |
| **Google Gemini** | https://aistudio.google.com/apikey | Yes — Flash/Flash-Lite models |

Cerebras was removed — all models now return 402 Payment Required on free keys. Every provider and model listed has been verified with a live API call.

---

## Architecture

```
app.py  (Streamlit UI)
│
├── sidebar
│   ├── Provider / model selection
│   ├── Session stats (tokens, elapsed time)
│   └── CSV + JSON export
│
├── chat display  (single mode or comparison mode)
│
└── chat input
    ├── Single-provider mode  ──► providers/openai_compatible.py  or  providers/gemini.py
    └── Comparison mode       ──► both providers in parallel columns

providers/
├── __init__.py          re-exports all public symbols
├── registry.py          provider configs, model lists, build_provider()
├── base.py              LLMProvider ABC + LLMResponse dataclass
├── openai_compatible.py Groq / Cerebras / OpenRouter adapter
└── gemini.py            Google Gemini adapter
```

---

## Component Map

### `providers/registry.py`

- `OPENAI_COMPATIBLE_CONFIGS` — base URL, env key, verified free model list for Groq and OpenRouter
- `NATIVE_PROVIDER_CONFIGS` — env key, model list for Google Gemini
- `build_provider(provider_name, model_name)` — factory; returns `LLMProvider` instance
- `get_models(provider_name)` — returns model list; OpenRouter fetches live from API (`:free` filter), others use hardcoded verified lists
- `has_api_key(provider_name)` — checks env var is present and not a placeholder value
- `_fetch_openrouter_models()` — `lru_cache`-wrapped live fetch from `https://openrouter.ai/api/v1/models`

### `providers/base.py`

```python
@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    model_used: str
    error: Optional[str]

    @property
    def total_tokens(self) -> int: ...

class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict]) -> LLMResponse: ...
```

### `providers/openai_compatible.py`

Adapter for any provider that speaks the OpenAI Chat Completions wire format. Lazy-loads the `openai` client on first use. Covers Groq and OpenRouter.

### `providers/gemini.py`

Adapter for Google Gemini using the `google-genai` SDK. Converts OpenAI-style message list to Gemini contents format, handles system messages separately.

### `app.py`

Session state keys:

| Key | Purpose |
|---|---|
| `chat_history` | All messages `[{"role": ..., "content": ...}]` |
| `message_reports` | One dict per assistant message with provider, model, token counts, elapsed |
| `session_stats` | Running totals: input/output tokens, elapsed, message count |
| `comparison_history_1` | Per-provider conversation context for comparison mode, provider 1 |
| `comparison_history_2` | Per-provider conversation context for comparison mode, provider 2 |
| `last_comparison_mode` | Detects mode switch to reset comparison histories |

---

## Data Flow

### Single-provider mode

```
user types  →  chat_history.append(user)
             →  build full messages[] from chat_history
             →  provider.chat(messages)
             →  chat_history.append(assistant)
             →  message_reports.append(report)
             →  session_stats updated
             →  st.rerun()
```

### Comparison mode

```
user types  →  chat_history.append(user)
             →  messages_1 = comparison_history_1 + [user]   # full context per provider
             →  messages_2 = comparison_history_2 + [user]
             →  provider_1.chat(messages_1)  │  rendered in col1
             →  provider_2.chat(messages_2)  │  rendered in col2
             →  chat_history.append(assistant_1)
             →  chat_history.append(assistant_2)
             →  comparison_history_1 updated
             →  comparison_history_2 updated
             →  message_reports.append(report_1)
             →  message_reports.append(report_2)
             →  session_stats updated
             →  st.rerun()
```

History display in comparison mode groups turns as: user + 2 assistants → rendered side-by-side.

---

## UML

### Class Diagram

```
LLMProvider (ABC)
    + chat(messages: list[dict]) → LLMResponse
         ▲
         │
   ┌─────┴──────────────────────┐
   │                            │
OpenAICompatibleProvider    GeminiProvider
   - base_url                   - api_key
   - api_key                    - model
   - model
   - client (lazy OpenAI)
```

### Sequence — Single Provider

```
User → Streamlit: submit message
Streamlit → session_state: append user message
Streamlit → registry: build_provider(provider, model)
registry → OpenAICompatibleProvider: __init__
Streamlit → provider: chat(messages)
provider → External API: POST /chat/completions
External API → provider: LLMResponse
provider → Streamlit: LLMResponse
Streamlit → session_state: append assistant + report
Streamlit → User: st.rerun() → display history
```

### Sequence — Comparison Mode

```
User → Streamlit: submit message
Streamlit → session_state: append user message
Streamlit → registry: build_provider(provider_1, model_1)
Streamlit → registry: build_provider(provider_2, model_2)
Streamlit → provider_1: chat(messages_1)   [col1]
Streamlit → provider_2: chat(messages_2)   [col2]
Both APIs respond
Streamlit → session_state: append assistant_1, assistant_2
Streamlit → session_state: update comparison_history_1/2
Streamlit → User: st.rerun() → display side-by-side history
```

### State Machine

```
[IDLE]
  │ user submits message
  ▼
[CALL PROVIDER(S)]
  │ response received
  ▼
[UPDATE SESSION STATE]  ──► error? → show error, still update state
  │
  ▼
[ST.RERUN]
  │
  ▼
[DISPLAY HISTORY]
  │ user submits next message
  ▼
  (loop)
```

---

## Verified Free Models

### Groq
- `llama-3.3-70b-versatile`
- `llama-3.1-8b-instant`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`

### OpenRouter
Fetched live from `https://openrouter.ai/api/v1/models` — only models with `:free` suffix are shown. Falls back to a hardcoded list (verified 2026-08-17) if the API is unreachable:
- `nvidia/nemotron-3-super-120b-a12b:free`
- `nvidia/nemotron-3.5-lightning:free`
- `nvidia/nemotron-nano-9b-v2:free`
- `google/gemma-4-31b-it:free`
- `openai/gpt-oss-20b:free`

### Google Gemini
- `gemini-flash-lite-latest`
- `gemini-flash-latest`

`gemini-pro-latest` is excluded — it maps to Gemini 3.1 Pro which has no free quota.

---

## Setup

```bash
cd FreeTierAI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your API keys
```

### `.env`

```
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-v1-...
GEMINI_API_KEY=...
```

Only providers with a valid (non-placeholder) API key will show ✅ in the sidebar. Others show ❌ and are still selectable but will return an auth error.

---

## Running

```bash
source venv/bin/activate
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Features

- **Single-provider chat** — full multi-turn conversation with context passed on every call
- **Comparison mode** — same prompt sent to two providers simultaneously, rendered side-by-side; each provider maintains its own conversation history for multi-turn use
- **Per-message metadata** — model ID, input→output token counts, and response time shown under each assistant message; comparison mode uses a compact single-line caption per provider
- **Session stats** — running totals in sidebar: input tokens, output tokens, elapsed time, message count
- **CSV export** — all messages + metadata in one file
- **JSON export** — full session data for analysis
- **Safety guardrails** — basic profanity/abuse filter on user input and model output
- **Clear chat** — resets all history, reports, stats, and comparison histories

---

## Safety Guardrails

User messages and model responses are checked against a word-list regex before display. Blocked messages show `[Blocked by safety guardrail: inappropriate language detected.]` instead of the original text. This is a basic first-pass filter, not a comprehensive content policy.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | UI framework |
| `openai` | OpenAI-compatible client (Groq, OpenRouter) |
| `google-genai` | Google Gemini SDK |
| `requests` | OpenRouter live model fetch |
| `python-dotenv` | `.env` file loading |
| `pandas` | (transitive, used by Streamlit) |

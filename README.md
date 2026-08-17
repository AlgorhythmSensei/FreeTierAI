# FreeTierAI — LLM Provider Comparison Tool

A Streamlit chatbot for comparing free-tier LLM providers side by side. Send prompts to different providers and models, track token usage and response times, and export session data for analysis.

## Features

### 🤖 Multi-Provider Support
- **OpenAI-Compatible Free-Tier Providers** (6 providers using one adapter):
  - [Groq](https://groq.com) — Ultra-fast inference on open models
  - [Cerebras](https://cerebras.ai) — On-chip LLM inference
  - [OpenRouter](https://openrouter.ai) — Unified API for 100+ models
  - [Mistral](https://mistral.ai) — Open and commercial models
   - [NVIDIA NIM](https://build.nvidia.com) — NVIDIA's inference microservices
   - [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) — 10,000 free Neurons per day

- **Native Free-Tier Providers**:
   - [Google Gemini](https://ai.google.dev/gemini-api/docs) — Free AI Studio/API tier with model-specific limits

### 📊 Detailed Metrics Per Message
- Input/output token counts
- Response time (elapsed seconds)
- Model used
- Error reporting

### 📈 Session Tracking
- Running totals in sidebar: total tokens, messages, cumulative time
- Per-message reports displayed below each response

### 💾 Session Export
- Download full chat history with all metrics as CSV
- Includes timestamp, provider, model, token counts, timing, errors
- Useful for building comparison charts afterward

### 🔄 Dynamic Model Lists *(Beta)*
- **Groq, OpenRouter, Mistral**: Fetch available models from provider API
- **Fallback**: If API call fails or provider doesn't support listing, use hardcoded curated list
- 3-second timeout per API call, so slow endpoints don't block the UI

### 🧠 Provider-Adapter Pattern
- Single `LLMProvider` abstract base class
- One adapter per provider type (or one class for OpenAI-compatible providers)
- Reusable `LLMResponse` dataclass for consistent metric reporting
- Easy to add new providers without duplicating code

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone or navigate to the project directory
cd FreeTierAI

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### API Keys
Set environment variables for the providers you want to use:

```bash
# OpenAI-compatible
export GROQ_API_KEY="your_groq_key"
export CEREBRAS_API_KEY="your_cerebras_key"
export OPENROUTER_API_KEY="your_openrouter_key"
export MISTRAL_API_KEY="your_mistral_key"
export NVIDIA_API_KEY="your_nvidia_key"
export CLOUDFLARE_API_TOKEN="your_cloudflare_api_token"
export CLOUDFLARE_ACCOUNT_ID="your_cloudflare_account_id"

# Google Gemini
export GEMINI_API_KEY="your_gemini_key"

```

Or create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key
CEREBRAS_API_KEY=your_cerebras_key
OPENROUTER_API_KEY=your_openrouter_key
MISTRAL_API_KEY=your_mistral_key
NVIDIA_API_KEY=your_nvidia_key
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
GEMINI_API_KEY=your_gemini_key
```

And load them with:
```bash
source .env  # or use python-dotenv in the app
```

---

## Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` by default.

---

## Project Structure

```
FreeTierAI/
├── providers/
│   ├── __init__.py                      # Public API
│   ├── base.py                          # LLMProvider abstract class + LLMResponse dataclass
│   ├── registry.py                      # Provider configs, dynamic model fetching, builder functions
│   ├── openai_compatible.py             # Adapter for compatible providers
│   └── gemini.py                         # Adapter for Google Gemini
├── app.py                               # Streamlit UI
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## Architecture

### Provider-Adapter Pattern

Each provider implements the `LLMProvider` abstract base class:

```python
class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict]) -> LLMResponse:
        """Send a chat request, return LLMResponse with metrics."""
        pass
```

All responses are normalized to the same `LLMResponse` dataclass:

```python
@dataclass
class LLMResponse:
    text: str                  # The actual response text
    input_tokens: int          # Prompt tokens used
    output_tokens: int         # Completion tokens used
    elapsed_seconds: float     # Wall-clock time
    model_used: str            # Actual model called
    error: Optional[str] = None
```

### Registry & Builder

`providers/registry.py` is the single source of truth:
- `OPENAI_COMPATIBLE_CONFIGS` — Config dict for 6 OpenAI-compatible providers
- `NATIVE_PROVIDER_CONFIGS` — Config dict for Google Gemini
- `get_models(provider_name)` — Fetch models dynamically (Groq, OpenRouter, Mistral) or return hardcoded list
- `build_provider(provider_name, model_name)` — Factory function to instantiate a provider
- `get_website(provider_name)` — Get the console/key management URL
- `missing_key_for(provider_name)` — Check if API key is set

### Dynamic Model Listing

For providers that expose a `/models` endpoint:

1. **Groq** (`/models` at `https://api.groq.com/openai/v1/models`):
   - Requires `GROQ_API_KEY`
   - Cached for the session (LRU cache, max 8 calls)
   - 3-second timeout; falls back to hardcoded list if unavailable

2. **OpenRouter** (`/models` at `https://openrouter.ai/api/v1/models`):
   - Public endpoint, no auth required
   - Returns 100+ models; app returns top 20
   - Cached and falls back to hardcoded list on error

3. **Mistral** (`/models` at `https://api.mistral.ai/v1/models`):
   - Requires `MISTRAL_API_KEY`
   - Cached for the session
   - 3-second timeout; falls back to hardcoded list

**Other providers** use hardcoded, curated model lists:
- Cerebras, NVIDIA NIM, Cloudflare Workers AI, and Google Gemini

---

## Usage

### Basic Chat

1. Open the app
2. Select a provider from the dropdown in the sidebar
3. Verify your API key is set (green checkmark)
4. Select a model
5. Type your message and press Enter
6. See the response, token usage, and timing below each message
7. Running totals appear in the sidebar

### Exporting Session Data

1. Click **📥 Export Session as CSV** in the sidebar
2. Click **💾 Download CSV** to save the file
3. Includes columns: timestamp, provider, model, input_tokens, output_tokens, elapsed_seconds, error

### Clearing Chat History

1. Click **🗑️ Clear Chat** to reset the conversation, stats, and message reports
2. Session state is cleared; you can start a fresh comparison

---

## Extending the Project

### Adding a New Provider

1. **Implement `LLMProvider`**:
   ```python
   from providers.base import LLMProvider, LLMResponse
   
   class MyProviderAdapter(LLMProvider):
       def __init__(self, api_key: str, model: str):
           self.api_key = api_key
           self.model = model
       
       def chat(self, messages: list[dict]) -> LLMResponse:
           # Call API, return LLMResponse
           pass
   ```

2. **Add config to `registry.py`**:
   - Add an entry to `OPENAI_COMPATIBLE_CONFIGS` for a provider that uses the OpenAI wire format
   - Include `env_key`, `website`, `models` list

3. **Update `build_provider()` in `registry.py`** to instantiate your adapter

4. **(Optional) Add dynamic model listing** if the provider has a `/models` endpoint:
   - Create a `_fetch_myprovider_models()` function
   - Decorate with `@lru_cache(maxsize=8)`
   - Update `get_models()` to call it

### Adding New Features

All features should follow the existing patterns:
- Use `build_provider()` and `LLMResponse` (don't create new abstractions)
- Store session state in `st.session_state`
- Keep provider logic in `providers/`, UI logic in `app.py`

---

## Limitations & Future Work

### Current Limitations
- **Model lists** are cached in-session; refresh by restarting the app
- **No multi-turn context aware responses yet** for some providers (basic message history is stored, but not all providers handle it identically)
- **Cloudflare Workers AI** requires both a Workers AI API token and account ID; the free allocation is limited to 10,000 Neurons per day
- **Google Gemini** free-tier availability and rate limits vary by model; paid-tier models are intentionally excluded

### Planned Features
- ✅ Dynamic model listing (in progress)
- ⬜ Side-by-side mode: send same prompt to 2 providers, compare responses
- ⬜ Streaming responses for faster feedback
- ⬜ Cost estimation based on token usage

---

## Troubleshooting

### "API key missing" error
- Check that your environment variable is set: `echo $GROQ_API_KEY`
- Restart the Streamlit app after setting the variable
- Use `.env` file if you prefer not to set system environment variables

### Slow model dropdown load
- First load may be slow if fetching from provider API (3-second timeout max)
- Subsequent loads are cached for the session
- If it times out, the hardcoded list is used instead

### "Cannot connect to provider" error
- Check your internet connection
- Verify the API key is correct
- Check if the provider's API is down
- Try a different provider to isolate the issue

### "No models available"
- This provider may not be configured yet; add models to `OPENAI_COMPATIBLE_CONFIGS` in `registry.py`
- If using a dynamic endpoint, verify the API key is set and the endpoint is accessible

---

## Dependencies

- **streamlit** — Web UI framework
- **openai** — Client SDK for the OpenAI-compatible providers via `base_url` override
- **google-genai** — Google Gemini API client
- **pandas** — Data handling (for CSV export)
- **requests** — HTTP library (for dynamic model fetching)

---

## License

(Add your license here, e.g., MIT, Apache 2.0, etc.)

---

## Contributing

Contributions welcome! Areas of interest:
- New provider adapters
- Streaming response support
- Improved caching/performance
- Additional export formats (JSON, PDF, etc.)
- Testing & documentation improvements


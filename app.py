# app.py
"""FreeTierAI — LLM Provider Comparison Tool."""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import csv
import json
import io
import re
from dotenv import load_dotenv

from providers import (
    build_provider,
    get_models,
    get_website,
    missing_key_for,
    list_all_providers,
)
from providers.registry import refresh_env_from_file

# Load environment variables from .env file
load_dotenv(override=True)
refresh_env_from_file()


def get_provider_model_options(provider_name: str):
    """Return the current API key status and available model list for a provider."""
    refresh_env_from_file()
    key_missing = missing_key_for(provider_name)
    models = get_models(provider_name, refresh=True)
    return key_missing, models

# ---------------------------------------------------------------------------
# BASIC SAFETY GUARDRAILS
# ---------------------------------------------------------------------------

BAD_LANGUAGE_PATTERNS = [
    r"\b(fuck|fucking|shit|bullshit|damn|bitch|asshole|jerk|idiot|moron|slut|whore)\b",
    r"\b(dick|piss|cock|son of a bitch|motherfucker|retard|nazi|kike|spic)\b",
    r"\b(rape|kill yourself|suicide)\b",
]


def contains_bad_language(text: str) -> bool:
    """Return True if the message looks like it contains abusive or offensive language."""
    if not text:
        return False
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    for pattern in BAD_LANGUAGE_PATTERNS:
        if re.search(pattern, normalized):
            return True
    return False


def sanitize_chat_response(text: str) -> str:
    """Block abusive text before showing it to the user."""
    if contains_bad_language(text):
        return "[Blocked by safety guardrail: inappropriate language detected.]"
    return text

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FreeTierAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 FreeTierAI — LLM Provider Comparison")
st.markdown("Compare free-tier LLM providers side by side")

# ---------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_elapsed": 0.0,
        "message_count": 0,
    }

if "message_reports" not in st.session_state:
    st.session_state.message_reports = []

# Per-provider conversation histories for comparison mode (maintains context per provider)
if "comparison_history_1" not in st.session_state:
    st.session_state.comparison_history_1 = []

if "comparison_history_2" not in st.session_state:
    st.session_state.comparison_history_2 = []

# Track last mode to detect switches and reset comparison histories
if "last_comparison_mode" not in st.session_state:
    st.session_state.last_comparison_mode = False

# ---------------------------------------------------------------------------
# SIDEBAR CONFIG
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    # Toggle for comparison mode
    comparison_mode = st.checkbox("🔄 Compare Mode (2 providers)", value=False)

    if comparison_mode:
        st.info("Send the same prompt to two providers and compare responses side-by-side")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Provider 1")
            providers = list_all_providers()
            groq_index = providers.index("Groq") if "Groq" in providers else 0
            selected_provider_1 = st.selectbox("Provider", providers, key="provider_select_1", index=groq_index)

            if selected_provider_1:
                website = get_website(selected_provider_1)
                if website:
                    st.markdown(f"[🔗 API Console]({website})")

                key_missing, models = get_provider_model_options(selected_provider_1)
                if key_missing:
                    st.error("❌ Key missing")
                else:
                    st.success("✅ Key found")

                if models:
                    selected_model_1 = st.selectbox("Model", models, key="model_select_1")
                else:
                    st.error("No models available")
                    selected_model_1 = None
            else:
                selected_model_1 = None

        with col2:
            st.subheader("Provider 2")
            provider_2_default_index = 1 if len(providers) > 1 else 0
            if "Groq" in providers:
                provider_2_default_index = 1 if providers[0] == "Groq" else 0
            selected_provider_2 = st.selectbox("Provider ", providers, key="provider_select_2", index=provider_2_default_index)

            if selected_provider_2:
                website = get_website(selected_provider_2)
                if website:
                    st.markdown(f"[🔗 API Console]({website})")

                key_missing, models = get_provider_model_options(selected_provider_2)
                if key_missing:
                    st.error("❌ Key missing")
                else:
                    st.success("✅ Key found")

                if models:
                    selected_model_2 = st.selectbox("Model ", models, key="model_select_2")
                else:
                    st.error("No models available")
                    selected_model_2 = None
            else:
                selected_model_2 = None

        selected_provider = selected_provider_1
        selected_model = selected_model_1

    else:
        st.subheader("Provider")
        providers = list_all_providers()
        groq_index = providers.index("Groq") if "Groq" in providers else 0
        selected_provider = st.selectbox("Select Provider", providers, key="provider_select", index=groq_index)

        if selected_provider:
            website = get_website(selected_provider)
            if website:
                st.markdown(f"[🔗 API Console]({website})")

            key_missing, models = get_provider_model_options(selected_provider)
            if key_missing:
                st.error(f"❌ API key missing for {selected_provider}")
            else:
                st.success(f"✅ API key found")

            if models:
                selected_model = st.selectbox("Select Model", models, key="model_select")
            else:
                st.error("No models available for this provider")
                selected_model = None
        else:
            selected_model = None

        selected_provider_1 = selected_provider
        selected_model_1 = selected_model
        selected_provider_2 = None
        selected_model_2 = None

    st.divider()

    # Session Stats
    st.header("📊 Session Stats")
    stats = st.session_state.session_stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Input Tokens", stats["total_input_tokens"])
        st.metric("Messages", stats["message_count"])
    with col2:
        st.metric("Output Tokens", stats["total_output_tokens"])
        st.metric("Elapsed (s)", f"{stats['total_elapsed']:.2f}")

    st.divider()

    # Export & Clear
    st.subheader("💾 Export & Actions")

    if st.session_state.chat_history:
        # Prepare CSV data
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)

        # Write header
        writer.writerow([
            "Message #",
            "Role",
            "Content",
            "Provider",
            "Model",
            "Input Tokens",
            "Output Tokens",
            "Total Tokens",
            "Elapsed (s)",
            "Error"
        ])

        # Write messages with their reports — count assistant messages seen so far
        assistant_count = 0
        for i, msg in enumerate(st.session_state.chat_history):
            report = None
            if msg["role"] == "assistant":
                if assistant_count < len(st.session_state.message_reports):
                    report = st.session_state.message_reports[assistant_count]
                assistant_count += 1

            content_preview = msg["content"][:100].replace("\n", " ")

            if report:
                writer.writerow([
                    i + 1,
                    msg["role"],
                    content_preview,
                    report.get("provider", "N/A"),
                    report.get("model", "N/A"),
                    report.get("input_tokens", 0),
                    report.get("output_tokens", 0),
                    report.get("input_tokens", 0) + report.get("output_tokens", 0),
                    f"{report.get('elapsed_seconds', 0):.3f}",
                    report.get("error", ""),
                ])
            else:
                writer.writerow([
                    i + 1,
                    msg["role"],
                    content_preview,
                    "", "", "", "", "", "", ""
                ])

        csv_data = csv_buffer.getvalue()

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 CSV (Chat + Metrics)",
                data=csv_data,
                file_name=f"freetierai_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            # JSON export for detailed analysis
            session_json = {
                "exported_at": datetime.now().isoformat(),
                "summary": {
                    "total_messages": len(st.session_state.chat_history),
                    "total_input_tokens": stats["total_input_tokens"],
                    "total_output_tokens": stats["total_output_tokens"],
                    "total_elapsed_seconds": stats["total_elapsed"],
                },
                "chat_history": st.session_state.chat_history,
                "reports": st.session_state.message_reports,
            }
            json_data = json.dumps(session_json, indent=2)
            st.download_button(
                label="📋 JSON (Full Data)",
                data=json_data,
                file_name=f"freetierai_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.info("💡 Start a conversation to enable export")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.message_reports = []
        st.session_state.comparison_history_1 = []
        st.session_state.comparison_history_2 = []
        st.session_state.session_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_elapsed": 0.0,
            "message_count": 0,
        }
        st.rerun()

# ---------------------------------------------------------------------------
# MODE SWITCH DETECTION — reset comparison histories when toggling modes
# ---------------------------------------------------------------------------

if comparison_mode != st.session_state.last_comparison_mode:
    st.session_state.comparison_history_1 = []
    st.session_state.comparison_history_2 = []
    st.session_state.last_comparison_mode = comparison_mode

# ---------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# ---------------------------------------------------------------------------

if comparison_mode:
    st.header("🔄 Provider Comparison")
else:
    st.header("💬 Chat")

# Display chat history
if not comparison_mode:
    # Single-provider mode: display sequentially
    assistant_count = 0
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg["role"] == "assistant":
                if assistant_count < len(st.session_state.message_reports):
                    report = st.session_state.message_reports[assistant_count]
                    if report.get("error"):
                        st.error(f"❌ {report['error']}")
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.caption(f"**Provider:** {report.get('provider', 'N/A')}")
                        with col2:
                            st.caption(f"**Model:** {report.get('model', 'N/A')}")
                        with col3:
                            st.caption(f"**Tokens:** {report.get('input_tokens', 0)}→{report.get('output_tokens', 0)}")
                        with col4:
                            st.caption(f"**Time:** {report.get('elapsed_seconds', 0):.2f}s")
                assistant_count += 1

else:
    # Comparison mode: group history into turns (user + 2 assistant messages per turn)
    history = st.session_state.chat_history
    reports = st.session_state.message_reports
    i = 0
    report_idx = 0
    while i < len(history):
        if history[i]["role"] == "user":
            with st.chat_message("user"):
                st.write(history[i]["content"])

            asst_1 = history[i + 1] if i + 1 < len(history) and history[i + 1]["role"] == "assistant" else None
            asst_2 = history[i + 2] if i + 2 < len(history) and history[i + 2]["role"] == "assistant" else None
            r1 = reports[report_idx] if report_idx < len(reports) else None
            r2 = reports[report_idx + 1] if report_idx + 1 < len(reports) else None

            col1, col2 = st.columns(2)
            if asst_1:
                with col1:
                    provider_label = r1.get("provider", "Provider 1") if r1 else "Provider 1"
                    st.markdown(f"**{provider_label}**")
                    with st.chat_message("assistant"):
                        st.write(asst_1["content"])
                        if r1:
                            if r1.get("error"):
                                st.error(f"❌ {r1['error']}")
                            else:
                                st.caption(f"**{r1.get('model', 'N/A')}** · {r1.get('input_tokens', 0)}→{r1.get('output_tokens', 0)} tokens · {r1.get('elapsed_seconds', 0):.2f}s")
            if asst_2:
                with col2:
                    provider_label = r2.get("provider", "Provider 2") if r2 else "Provider 2"
                    st.markdown(f"**{provider_label}**")
                    with st.chat_message("assistant"):
                        st.write(asst_2["content"])
                        if r2:
                            if r2.get("error"):
                                st.error(f"❌ {r2['error']}")
                            else:
                                st.caption(f"**{r2.get('model', 'N/A')}** · {r2.get('input_tokens', 0)}→{r2.get('output_tokens', 0)} tokens · {r2.get('elapsed_seconds', 0):.2f}s")

            step = 1 + (1 if asst_1 else 0) + (1 if asst_2 else 0)
            i += step
            report_idx += (1 if r1 else 0) + (1 if r2 else 0)
        else:
            i += 1

# User input
user_input = st.chat_input("Enter your message...", key="user_input")

if user_input:
    if contains_bad_language(user_input):
        st.warning("⚠️ Your message contains inappropriate language. Please rephrase it respectfully.")
    elif comparison_mode:
        # Side-by-side mode
        if not selected_provider_1 or not selected_model_1 or not selected_provider_2 or not selected_model_2:
            st.error("Please select both providers and models")
        else:
            # Build per-provider message lists including full conversation history
            messages_1 = list(st.session_state.comparison_history_1) + [{"role": "user", "content": user_input}]
            messages_2 = list(st.session_state.comparison_history_2) + [{"role": "user", "content": user_input}]

            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
            })

            # Initialize report variables before try blocks
            report_1 = None
            report_2 = None
            response_1 = None
            response_2 = None

            # Call both providers
            st.subheader("📊 Comparison Results")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {selected_provider_1}")
                try:
                    provider_1 = build_provider(selected_provider_1, selected_model_1)
                    response_1 = provider_1.chat(messages_1)

                    if response_1.error:
                        st.error(f"❌ {response_1.error}")
                    else:
                        response_1.text = sanitize_chat_response(response_1.text)
                        if response_1.text.startswith("[Blocked by safety guardrail"):
                            st.warning(response_1.text)
                        else:
                            st.write(response_1.text)

                        # Show metrics
                        mc1, mc2 = st.columns(2)
                        with mc1:
                            st.metric("Tokens (in→out)", f"{response_1.input_tokens}→{response_1.output_tokens}")
                        with mc2:
                            st.metric("Time (s)", f"{response_1.elapsed_seconds:.3f}")

                        report_1 = {
                            "timestamp": datetime.now().isoformat(),
                            "provider": selected_provider_1,
                            "model": selected_model_1,
                            "input_tokens": response_1.input_tokens,
                            "output_tokens": response_1.output_tokens,
                            "elapsed_seconds": response_1.elapsed_seconds,
                            "error": response_1.error,
                        }
                except Exception as e:
                    st.error(f"Error: {str(e)}")

            with col2:
                st.markdown(f"### {selected_provider_2}")
                try:
                    provider_2 = build_provider(selected_provider_2, selected_model_2)
                    response_2 = provider_2.chat(messages_2)

                    if response_2.error:
                        st.error(f"❌ {response_2.error}")
                    else:
                        response_2.text = sanitize_chat_response(response_2.text)
                        if response_2.text.startswith("[Blocked by safety guardrail"):
                            st.warning(response_2.text)
                        else:
                            st.write(response_2.text)

                        # Show metrics
                        mc1, mc2 = st.columns(2)
                        with mc1:
                            st.metric("Tokens (in→out)", f"{response_2.input_tokens}→{response_2.output_tokens}")
                        with mc2:
                            st.metric("Time (s)", f"{response_2.elapsed_seconds:.3f}")

                        report_2 = {
                            "timestamp": datetime.now().isoformat(),
                            "provider": selected_provider_2,
                            "model": selected_model_2,
                            "input_tokens": response_2.input_tokens,
                            "output_tokens": response_2.output_tokens,
                            "elapsed_seconds": response_2.elapsed_seconds,
                            "error": response_2.error,
                        }
                except Exception as e:
                    st.error(f"Error: {str(e)}")

            # Append assistant messages to chat_history and update per-provider histories
            if response_1 is not None:
                text_1 = response_1.text if not response_1.error else f"[Error: {response_1.error}]"
                st.session_state.chat_history.append({"role": "assistant", "content": text_1})
                st.session_state.comparison_history_1.append({"role": "user", "content": user_input})
                if not response_1.error:
                    st.session_state.comparison_history_1.append({"role": "assistant", "content": response_1.text})

            if response_2 is not None:
                text_2 = response_2.text if not response_2.error else f"[Error: {response_2.error}]"
                st.session_state.chat_history.append({"role": "assistant", "content": text_2})
                st.session_state.comparison_history_2.append({"role": "user", "content": user_input})
                if not response_2.error:
                    st.session_state.comparison_history_2.append({"role": "assistant", "content": response_2.text})

            # Store both reports and update stats
            if report_1 is not None:
                st.session_state.message_reports.append(report_1)
                if response_1 is not None and not response_1.error:
                    st.session_state.session_stats["total_input_tokens"] += response_1.input_tokens
                    st.session_state.session_stats["total_output_tokens"] += response_1.output_tokens
                    st.session_state.session_stats["total_elapsed"] += response_1.elapsed_seconds

            if report_2 is not None:
                st.session_state.message_reports.append(report_2)
                if response_2 is not None and not response_2.error:
                    st.session_state.session_stats["total_input_tokens"] += response_2.input_tokens
                    st.session_state.session_stats["total_output_tokens"] += response_2.output_tokens
                    st.session_state.session_stats["total_elapsed"] += response_2.elapsed_seconds

            st.session_state.session_stats["message_count"] += 1  # One user turn

            # Comparison summary
            st.divider()
            st.markdown("### ⚡ Quick Comparison")
            comp_col1, comp_col2, comp_col3 = st.columns(3)

            if response_1 is not None and response_2 is not None and not response_1.error and not response_2.error:
                total_1 = response_1.input_tokens + response_1.output_tokens
                total_2 = response_2.input_tokens + response_2.output_tokens
                time_diff = abs(response_1.elapsed_seconds - response_2.elapsed_seconds)
                faster = selected_provider_1 if response_1.elapsed_seconds < response_2.elapsed_seconds else selected_provider_2

                with comp_col1:
                    st.write(f"**Faster:** {faster}")
                    st.write(f"Difference: {time_diff:.3f}s")

                with comp_col2:
                    st.write(f"**Token Efficiency:**")
                    st.write(f"{selected_provider_1}: {total_1} | {selected_provider_2}: {total_2}")

                with comp_col3:
                    quality_note = "Similar outputs" if abs(len(response_1.text) - len(response_2.text)) < 100 else "Different lengths"
                    st.write(f"**Response Length:**")
                    st.write(f"{quality_note}")

            st.rerun()

    elif selected_provider and selected_model:
        # Single-provider mode
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
        })

        # Build message list for API
        messages = []
        for msg in st.session_state.chat_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        # Call provider
        try:
            provider = build_provider(selected_provider, selected_model)
            response = provider.chat(messages)
            response.text = sanitize_chat_response(response.text)

            # Add assistant message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.text,
            })

            # Record report
            report = {
                "timestamp": datetime.now().isoformat(),
                "provider": selected_provider,
                "model": selected_model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "elapsed_seconds": response.elapsed_seconds,
                "error": response.error,
            }
            st.session_state.message_reports.append(report)

            # Update session stats
            if not response.error:
                st.session_state.session_stats["total_input_tokens"] += response.input_tokens
                st.session_state.session_stats["total_output_tokens"] += response.output_tokens
                st.session_state.session_stats["total_elapsed"] += response.elapsed_seconds
            st.session_state.session_stats["message_count"] += 1

        except Exception as e:
            st.error(f"Error: {str(e)}")

        st.rerun()

    else:
        if not selected_provider:
            st.error("Please select a provider")
        if not selected_model:
            st.error("Please select a model")

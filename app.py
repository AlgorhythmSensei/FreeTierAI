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
        
        # Write messages with their reports
        for i, msg in enumerate(st.session_state.chat_history):
            report = None
            if msg["role"] == "assistant" and i < len(st.session_state.message_reports):
                # Find the corresponding report (skip user messages)
                report_idx = sum(1 for m in st.session_state.chat_history[:i] if m["role"] == "assistant") - 1
                if 0 <= report_idx < len(st.session_state.message_reports):
                    report = st.session_state.message_reports[report_idx]
            
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
        st.session_state.session_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_elapsed": 0.0,
            "message_count": 0,
        }
        st.rerun()

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
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Show report if available
            if i < len(st.session_state.message_reports):
                report = st.session_state.message_reports[i]
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
            # Build message list
            messages = [{"role": "user", "content": user_input}]
            
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
            })
            
            # Call both providers
            st.subheader("📊 Comparison Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {selected_provider_1}")
                try:
                    provider_1 = build_provider(selected_provider_1, selected_model_1)
                    response_1 = provider_1.chat(messages)
                    
                    if response_1.error:
                        st.error(f"❌ {response_1.error}")
                    else:
                        response_1.text = sanitize_chat_response(response_1.text)
                        if response_1.text.startswith("[Blocked by safety guardrail"):
                            st.warning(response_1.text)
                        else:
                            st.write(response_1.text)
                        
                        # Show metrics
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric("Input Tokens", response_1.input_tokens)
                        with metric_col2:
                            st.metric("Output Tokens", response_1.output_tokens)
                        with metric_col3:
                            st.metric("Total Tokens", response_1.total_tokens)
                        with metric_col4:
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
                    response_1 = None
                    report_1 = None
            
            with col2:
                st.markdown(f"### {selected_provider_2}")
                try:
                    provider_2 = build_provider(selected_provider_2, selected_model_2)
                    response_2 = provider_2.chat(messages)
                    
                    if response_2.error:
                        st.error(f"❌ {response_2.error}")
                    else:
                        response_2.text = sanitize_chat_response(response_2.text)
                        if response_2.text.startswith("[Blocked by safety guardrail"):
                            st.warning(response_2.text)
                        else:
                            st.write(response_2.text)
                        
                        # Show metrics
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric("Input Tokens", response_2.input_tokens)
                        with metric_col2:
                            st.metric("Output Tokens", response_2.output_tokens)
                        with metric_col3:
                            st.metric("Total Tokens", response_2.total_tokens)
                        with metric_col4:
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
                    response_2 = None
                    report_2 = None
            
            # Store both reports
            if report_1:
                st.session_state.message_reports.append(report_1)
                if not response_1.error:
                    st.session_state.session_stats["total_input_tokens"] += response_1.input_tokens
                    st.session_state.session_stats["total_output_tokens"] += response_1.output_tokens
                    st.session_state.session_stats["total_elapsed"] += response_1.elapsed_seconds
            
            if report_2:
                st.session_state.message_reports.append(report_2)
                if not response_2.error:
                    st.session_state.session_stats["total_input_tokens"] += response_2.input_tokens
                    st.session_state.session_stats["total_output_tokens"] += response_2.output_tokens
                    st.session_state.session_stats["total_elapsed"] += response_2.elapsed_seconds
            
            st.session_state.session_stats["message_count"] += 2  # Two responses
            
            # Comparison summary
            st.divider()
            st.markdown("### ⚡ Quick Comparison")
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            if response_1 and response_2:
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

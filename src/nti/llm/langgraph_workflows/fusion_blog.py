"""LangGraph Multi-LLM Fusion Workflow for Blog Generation.

Architecture:
    1. FAN-OUT: Multiple LLMs generate blog drafts in parallel (using Send() API)
    2. CRITIQUE: Each model critiques the others' outputs (cross-criticism)
    3. SYNTHESIZE: The most powerful available model merges all drafts + critiques
       into a single final blog post

This implements the "fusion of models" strategy where:
- Multiple providers (Groq, Gemini, Cerebras, OpenRouter, Together, Nvidia, etc.)
  are called in parallel using OpenAI-compatible endpoints
- Each provider uses ChatOpenAI with a different base_url
- All providers use the same chat/completions format (OpenAI-compatible)
- The system automatically falls back if a provider fails
- Everything is configurable via .env (enable/disable each provider)
"""

from __future__ import annotations

import logging
import os
import time
from typing import TypedDict, Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from nti.config.settings import settings, LLMProviderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Langfuse Observability (optional — traces LLM calls for debugging/monitoring)
# ---------------------------------------------------------------------------


def _get_langfuse_handler():
    """Get a Langfuse callback handler if Langfuse is enabled and configured.

    Langfuse v4+ CallbackHandler reads credentials from environment variables:
      LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    We set these from our settings before creating the handler.

    Returns:
        CallbackHandler instance or None if Langfuse is not enabled.
    """
    if not settings.langfuse_enabled:
        return None
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.debug("Langfuse enabled but keys not configured — skipping tracing")
        return None
    try:
        # Set env vars that Langfuse CallbackHandler reads internally
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_base_url)

        # Initialize the Langfuse client first (required for CallbackHandler to trace)
        from langfuse import Langfuse
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )

        from langfuse.langchain import CallbackHandler
        return CallbackHandler(public_key=settings.langfuse_public_key)
    except ImportError:
        logger.warning("langfuse/langchain not installed. Install with: uv add langfuse langchain. Skipping tracing.")
        return None
    except Exception as e:
        logger.warning(f"Langfuse handler init failed: {e}")
        return None


# ---------------------------------------------------------------------------
# State Schema — shared data flowing through the graph
# ---------------------------------------------------------------------------


class FusionState(TypedDict, total=False):
    """State flowing through the LangGraph fusion workflow.

    Each node reads from and writes to this shared state.
    """

    # Input
    prompt: str                       # The blog generation prompt with all data
    news_headlines: str               # Recent news headlines text
    system_prompt: str                # System instructions for the LLM

    # Fan-out: parallel generation results
    drafts: Annotated[list[dict], lambda a, b: a + b]  # [{provider, content, error}]
    draft_count: int                  # How many drafts we expect

    # Critique results
    critiques: Annotated[list[dict], lambda a, b: a + b]  # [{provider, target_provider, critique}]

    # Synthesis
    final_blog: str                   # The final synthesized blog post

    # Metadata
    providers_used: list[str]         # Which providers were actually called
    errors: Annotated[list[str], lambda a, b: a + b]  # Error messages
    start_time: float                 # For timing
    total_duration_seconds: float


class DraftState(TypedDict):
    """State for a single parallel draft generation (via Send)."""

    prompt: str
    system_prompt: str
    provider: LLMProviderConfig


class CritiqueState(TypedDict):
    """State for a single parallel critique (via Send)."""

    draft_content: str
    draft_provider: str
    critic_provider: LLMProviderConfig
    all_drafts_summary: str


# ---------------------------------------------------------------------------
# Helper: Create ChatOpenAI instance for a given provider
# ---------------------------------------------------------------------------


def _make_llm(provider: LLMProviderConfig) -> BaseChatModel:
    """Create an LLM instance configured for the given provider.

    Provider types:
      - "openai_compatible": Uses ChatOpenAI with base_url (Groq, Cerebras, etc.)
      - "cohere": Uses langchain_cohere.ChatCohere with native Cohere SDK
      - "huggingface": Uses langchain_huggingface.ChatHuggingFace with HF Inference API

    All providers return a LangChain BaseChatModel-compatible object.
    """
    provider_type = provider.provider_type

    if provider_type == "cohere":
        return _make_cohere_llm(provider)
    elif provider_type == "huggingface":
        return _make_huggingface_llm(provider)
    else:
        # Default: OpenAI-compatible (Groq, Gemini, Cerebras, OpenRouter, etc.)
        base_name = provider.name.split("/")[0]
        extra_headers = None
        if base_name == "openrouter":
            extra_headers = {"HTTP-Referer": "https://nifty-timing-index.com"}
        return ChatOpenAI(
            model=provider.model,
            api_key=provider.api_key,
            base_url=provider.base_url,
            max_tokens=provider.max_tokens,
            temperature=0.7,
            default_headers=extra_headers,
        )


def _make_cohere_llm(provider: LLMProviderConfig) -> BaseChatModel:
    """Create a Cohere LLM instance using the native SDK.

    Uses langchain_cohere.ChatCohere which provides proper Cohere API support.
    Falls back to ChatOpenAI with Cohere's OpenAI-compatible endpoint if
    langchain_cohere is not installed.
    """
    try:
        from langchain_cohere import ChatCohere
        return ChatCohere(
            model=provider.model,
            cohere_api_key=provider.api_key,
            max_tokens=provider.max_tokens,
            temperature=0.7,
        )
    except ImportError:
        logger.warning(
            "langchain_cohere not installed. Install with: uv add langchain-cohere "
            "Falling back to OpenAI-compatible endpoint."
        )
        # Cohere also supports an OpenAI-compatible endpoint at /v2
        return ChatOpenAI(
            model=provider.model,
            api_key=provider.api_key,
            base_url=provider.base_url or "https://api.cohere.com/v2",
            max_tokens=provider.max_tokens,
            temperature=0.7,
        )


def _make_huggingface_llm(provider: LLMProviderConfig) -> BaseChatModel:
    """Create a HuggingFace LLM instance using the native SDK.

    Uses langchain_huggingface.ChatHuggingFace for proper HF Inference API support.
    Falls back to ChatOpenAI with HF's OpenAI-compatible endpoint if
    langchain_huggingface is not installed.
    """
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        llm = HuggingFaceEndpoint(
            repo_id=provider.model,
            huggingfacehub_api_token=provider.api_key,
            max_new_tokens=provider.max_tokens,
            temperature=0.7,
        )
        return ChatHuggingFace(llm=llm)
    except ImportError:
        logger.warning(
            "langchain_huggingface not installed. Install with: uv add langchain-huggingface "
            "Falling back to OpenAI-compatible endpoint."
        )
        # HuggingFace Inference API also supports OpenAI-compatible format
        return ChatOpenAI(
            model=provider.model,
            api_key=provider.api_key,
            base_url=provider.base_url or "https://api-inference.huggingface.co/v1",
            max_tokens=provider.max_tokens,
            temperature=0.7,
        )


# ---------------------------------------------------------------------------
# Node 1: Initialize — set up the state
# ---------------------------------------------------------------------------


def initialize_node(state: FusionState) -> dict:
    """Initialize the fusion workflow state."""
    enabled_providers = settings.get_enabled_llm_providers()
    provider_names = [p.name for p in enabled_providers]

    if not enabled_providers:
        return {
            "errors": ["No LLM providers enabled or configured. Check .env settings."],
            "drafts": [],
            "critiques": [],
            "providers_used": [],
            "start_time": time.time(),
        }

    logger.info(f"Fusion workflow: {len(enabled_providers)} providers enabled: {provider_names}")
    return {
        "drafts": [],
        "critiques": [],
        "providers_used": provider_names,
        "draft_count": min(len(enabled_providers), settings.fusion_max_concurrent),
        "errors": [],
        "start_time": time.time(),
    }


# ---------------------------------------------------------------------------
# Node 2: Fan-out — dispatch parallel LLM generation calls via Send()
# ---------------------------------------------------------------------------


def fan_out_dispatch(state: FusionState) -> list[Send]:
    """Dispatch parallel generation calls to all enabled LLM providers.

    Uses LangGraph's Send() API for parallel execution (fan-out pattern).
    Each Send creates a separate execution of the 'generate_draft' node
    with a specific provider configuration.
    """
    enabled_providers = settings.get_enabled_llm_providers()
    # Limit concurrent calls
    providers_to_call = enabled_providers[: settings.fusion_max_concurrent]

    sends = []
    for provider in providers_to_call:
        sends.append(
            Send(
                "generate_draft",
                {
                    "prompt": state.get("prompt", ""),
                    "system_prompt": state.get("system_prompt", ""),
                    "provider": provider,
                },
            )
        )

    provider_display = [f"{p.name}({p.model})" for p in providers_to_call]
    logger.info(f"Fan-out: dispatching {len(sends)} parallel generation calls: {provider_display}")
    return sends


# ---------------------------------------------------------------------------
# Node 3: Generate Draft — called in parallel for each provider
# ---------------------------------------------------------------------------


import threading

# Global state to stagger requests to the same provider
PROVIDER_LOCKS = {}
PROVIDER_LAST_CALLED = {}
DELAY_BETWEEN_CALLS = 8.0  # Seconds to wait between calls to the SAME provider

def _wait_for_provider_rate_limit(provider_base_name: str):
    """Ensure API calls to the same provider are staggered to avoid rate limits."""
    if provider_base_name not in PROVIDER_LOCKS:
        # Initialize lock for this provider if it doesn't exist
        # This is safe enough for typical LangGraph thread pools
        PROVIDER_LOCKS[provider_base_name] = threading.Lock()
    
    with PROVIDER_LOCKS[provider_base_name]:
        now = time.time()
        last_called = PROVIDER_LAST_CALLED.get(provider_base_name, 0)
        elapsed = now - last_called
        if elapsed < DELAY_BETWEEN_CALLS:
            sleep_time = DELAY_BETWEEN_CALLS - elapsed
            logger.info(f"Staggering {provider_base_name} request by {sleep_time:.1f}s to prevent rate limits")
            time.sleep(sleep_time)
        PROVIDER_LAST_CALLED[provider_base_name] = time.time()


def generate_draft(state: DraftState) -> dict:
    """Generate a blog draft using a single LLM provider.

    This node is invoked in parallel (via Send) for each enabled provider.
    Each call is independent and can succeed or fail without affecting others.
    """
    provider = state["provider"]
    prompt = state["prompt"]
    system_prompt = state.get("system_prompt", "")

    # Extract base provider name (e.g. "openrouter" from "openrouter/model-name")
    provider_base = provider.name.split("/")[0].lower()
    
    logger.info(f"Queued draft generation with provider: {provider.name} (model: {provider.model})")

    try:
        # Wait if we recently called this SAME provider
        _wait_for_provider_rate_limit(provider_base)
        
        logger.info(f"Executing draft generation with provider: {provider.name}")
        llm = _make_llm(provider)
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = llm.invoke(messages)
        content = response.content

        logger.info(f"Draft from {provider.name}: {len(content)} chars generated")
        return {
            "drafts": [
                {
                    "provider": provider.name,
                    "model": provider.model,
                    "content": content,
                    "error": None,
                    "char_count": len(content),
                }
            ],
            "errors": [],
        }

    except Exception as e:
        error_msg = f"Provider {provider.name} ({provider.model}) failed: {e}"
        logger.warning(error_msg)
        return {
            "drafts": [
                {
                    "provider": provider.name,
                    "model": provider.model,
                    "content": "",
                    "error": str(e),
                    "char_count": 0,
                }
            ],
            "errors": [error_msg],
        }


# ---------------------------------------------------------------------------
# Node 4: Fan-out critique — dispatch parallel critique calls
# ---------------------------------------------------------------------------


def fan_out_critique_dispatch(state: FusionState) -> list[Send]:
    """Dispatch parallel critique calls.

    Each enabled provider critiques the drafts from OTHER providers.
    This implements the "criticize each other" strategy.

    With multi-model support (compound names like "groq/llama-3.3-70b"),
    models from the same provider can also critique each other when there
    are no drafts from other providers.
    """
    successful_drafts = [d for d in state.get("drafts", []) if d.get("content") and not d.get("error")]

    if len(successful_drafts) < 2:
        # Need at least 2 drafts to critique each other
        logger.info("Fewer than 2 successful drafts — skipping critique round")
        return []

    enabled_providers = settings.get_enabled_llm_providers()
    providers_to_call = enabled_providers[: settings.fusion_max_concurrent]

    # Build a summary of all drafts for context
    all_drafts_summary = "\n\n".join(
        f"--- Draft from {d['provider']} ({d['model']}) ---\n{d['content'][:1000]}..."
        for d in successful_drafts
    )

    sends = []
    for provider in providers_to_call:
        # Each provider critiques drafts from OTHER providers (by full name)
        other_drafts = [d for d in successful_drafts if d["provider"] != provider.name]
        # If no drafts from other providers, let this model critique other models
        # from the same provider (multi-model self-critique)
        # With compound names like "groq/model-a" vs "groq/model-b", they differ
        # and CAN critique each other. Only skip if this is the ONLY draft.
        if not other_drafts and len(successful_drafts) > 1:
            other_drafts = successful_drafts
        if not other_drafts:
            continue

        for target_draft in other_drafts:
            sends.append(
                Send(
                    "critique_draft",
                    {
                        "draft_content": target_draft["content"],
                        "draft_provider": target_draft["provider"],
                        "critic_provider": provider,
                        "all_drafts_summary": all_drafts_summary,
                    },
                )
            )

    # Limit total critiques to avoid rate limiting
    max_critiques = settings.fusion_critique_rounds * len(successful_drafts)
    sends = sends[:max_critiques]

    logger.info(f"Fan-out critique: dispatching {len(sends)} parallel critique calls")
    return sends


# ---------------------------------------------------------------------------
# Node 5: Critique Draft — called in parallel for each provider+draft pair
# ---------------------------------------------------------------------------


def critique_draft(state: CritiqueState) -> dict:
    """Have one LLM critique another LLM's draft.

    This implements the "models criticize each other" strategy.
    Each critic provides specific improvement suggestions.
    """
    critic_provider = state["critic_provider"]
    draft_content = state["draft_content"]
    draft_provider = state["draft_provider"]

    provider_base = critic_provider.name.split("/")[0].lower()
    logger.info(f"Queued critique: {critic_provider.name} critiquing {draft_provider}'s draft")

    critique_prompt = f"""You are a critical reviewer for a market analysis blog post.
Below is a draft blog post generated by another AI model ({draft_provider}).
Provide specific, constructive criticism focusing on:

1. **Accuracy**: Are the numbers and claims correct and precise?
2. **Completeness**: Is any important market data or context missing?
3. **Clarity**: Is the writing clear and jargon-free for Indian retail investors?
4. **Value**: Does it provide actionable insights for long-only value investors?
5. **PSU Context**: Does it properly highlight PSU/government stocks?
6. **Disclaimer**: Does it include the required disclaimer?

Keep your critique concise (200 words max). Focus on what should be IMPROVED.

--- DRAFT TO CRITIQUE ({draft_provider}) ---
{draft_content[:2000]}
--- END DRAFT ---
"""

    try:
        _wait_for_provider_rate_limit(provider_base)
        
        logger.info(f"Executing critique with provider: {critic_provider.name}")
        llm = _make_llm(critic_provider)
        response = llm.invoke([HumanMessage(content=critique_prompt)])

        return {
            "critiques": [
                {
                    "critic_provider": critic_provider.name,
                    "critic_model": critic_provider.model,
                    "target_provider": draft_provider,
                    "critique": response.content,
                }
            ],
            "errors": [],
        }

    except Exception as e:
        error_msg = f"Critique from {critic_provider.name} for {draft_provider} failed: {e}"
        logger.warning(error_msg)
        return {
            "critiques": [],
            "errors": [error_msg],
        }


# ---------------------------------------------------------------------------
# Node 6: Synthesize — merge all drafts + critiques into final blog
# ---------------------------------------------------------------------------


def synthesize_node(state: FusionState) -> dict:
    """Synthesize final blog post from all drafts and critiques.

    This is the fan-in node that uses the most powerful available model
    to merge all drafts, taking critiques into account.
    """
    successful_drafts = [d for d in state.get("drafts", []) if d.get("content") and not d.get("error")]
    critiques = state.get("critiques", [])

    if not successful_drafts:
        all_errors = state.get("errors", [])
        return {
            "final_blog": f"⚠️ Blog generation failed — no LLM providers succeeded. Errors: {'; '.join(all_errors)}",
            "total_duration_seconds": time.time() - state.get("start_time", time.time()),
        }

    # If only one draft succeeded, use it directly (with critique improvements if available)
    if len(successful_drafts) == 1:
        draft = successful_drafts[0]
        content = draft["content"]

        # If there are critiques, apply them via the synthesizer
        if critiques:
            content = _apply_critiques_single(draft, critiques, state)
        else:
            logger.info(f"Using single draft from {draft['provider']} (no critiques)")

        return {
            "final_blog": content,
            "total_duration_seconds": time.time() - state.get("start_time", time.time()),
        }

    # Multiple drafts — use synthesizer model to merge
    return _synthesize_multiple(successful_drafts, critiques, state)


def _apply_critiques_single(
    draft: dict,
    critiques: list[dict],
    state: FusionState,
) -> str:
    """Apply critiques to a single draft using the synthesizer model."""
    synthesizer = settings.get_synthesizer_provider()
    if not synthesizer:
        return draft["content"]

    logger.info(f"Applying {len(critiques)} critiques to {draft['provider']}'s draft via {synthesizer.name}")

    critique_text = "\n\n".join(
        f"Critique by {c['critic_provider']}:\n{c['critique']}" for c in critiques
    )

    synthesis_prompt = f"""You are NTI-Writer, the final editor for the Nifty Timing Index blog.
Take the draft below and improve it based on the critiques provided.
Keep the structure and data intact, but address each critique point.

--- ORIGINAL DRAFT ({draft['provider']}) ---
{draft['content']}

--- CRITIQUES ---
{critique_text}
--- END CRITIQUES ---

Produce the IMPROVED version of the blog post. Keep the same data and structure,
but fix any issues raised in the critiques. Include the disclaimer at the end.
"""

    try:
        llm = _make_llm(synthesizer)
        response = llm.invoke([HumanMessage(content=synthesis_prompt)])
        return response.content
    except Exception as e:
        logger.warning(f"Synthesizer failed, using original draft: {e}")
        return draft["content"]


def _synthesize_multiple(
    drafts: list[dict],
    critiques: list[dict],
    state: FusionState,
) -> dict:
    """Synthesize multiple drafts into a single final blog post."""
    synthesizer = settings.get_synthesizer_provider()
    if not synthesizer:
        # Fallback: use the longest draft
        best_draft = max(drafts, key=lambda d: d.get("char_count", 0))
        return {
            "final_blog": best_draft["content"],
            "total_duration_seconds": time.time() - state.get("start_time", time.time()),
        }

    logger.info(f"Synthesizing {len(drafts)} drafts via {synthesizer.name} ({synthesizer.model})")

    drafts_text = "\n\n".join(
        f"=== Draft from {d['provider']} ({d['model']}) ===\n{d['content']}\n=== End Draft ==="
        for d in drafts
    )

    critiques_text = ""
    if critiques:
        critiques_text = "\n\n=== CRITIQUES ===\n" + "\n\n".join(
            f"By {c['critic_provider']} on {c['target_provider']}'s draft:\n{c['critique']}"
            for c in critiques
        ) + "\n=== END CRITIQUES ==="

    synthesis_prompt = f"""You are NTI-Writer, the final synthesizer for the Nifty Timing Index blog.
You have received multiple draft blog posts from different AI models, plus cross-critiques.

Your task: Create ONE definitive blog post that:
1. Takes the BEST elements from each draft (most accurate data, clearest explanations)
2. Addresses all valid critiques raised by the models
3. Maintains consistency with the original prompt requirements
4. Keeps the Indian market context and value investing focus
5. Includes PSU stock mentions where relevant
6. Ends with the required disclaimer

Do NOT just concatenate the drafts. SYNTHESIZE them into one cohesive, high-quality post.

--- DRAFTS FROM MULTIPLE MODELS ---
{drafts_text}
{critiques_text}

Produce the FINAL, SYNTHESIZED blog post. It should be the best possible version
combining insights from all models.
"""

    try:
        llm = _make_llm(synthesizer)
        response = llm.invoke([HumanMessage(content=synthesis_prompt)])
        final_blog = response.content

        duration = time.time() - state.get("start_time", time.time())
        logger.info(f"Synthesis complete: {len(final_blog)} chars in {duration:.1f}s")

        return {
            "final_blog": final_blog,
            "total_duration_seconds": duration,
        }

    except Exception as e:
        logger.warning(f"Synthesizer failed ({e}), falling back to best draft")
        best_draft = max(drafts, key=lambda d: d.get("char_count", 0))
        return {
            "final_blog": best_draft["content"],
            "total_duration_seconds": time.time() - state.get("start_time", time.time()),
        }


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------


def build_fusion_graph() -> StateGraph:
    """Build the LangGraph multi-LLM fusion workflow.

    Graph structure:
        initialize → fan_out_dispatch → generate_draft (parallel) → fan_out_critique_dispatch
            → critique_draft (parallel) → synthesize → END

    If only 1 draft succeeds, critique is skipped.
    If no drafts succeed, the final_blog contains an error message.
    """
    graph = StateGraph(FusionState)

    # Add nodes
    graph.add_node("initialize", initialize_node)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("fan_out_critique_dispatch", fan_out_critique_dispatch)
    graph.add_node("critique_draft", critique_draft)
    graph.add_node("synthesize", synthesize_node)

    # Entry point
    graph.set_entry_point("initialize")

    # After initialization, fan out to parallel generation
    graph.add_conditional_edges("initialize", fan_out_dispatch, ["generate_draft"])

    # After all drafts generated, go to critique dispatch
    graph.add_edge("generate_draft", "fan_out_critique_dispatch")

    # After critiques, go to synthesize
    # If no critiques dispatched (fewer than 2 drafts), go straight to synthesize
    graph.add_conditional_edges(
        "fan_out_critique_dispatch",
        lambda state: "critique_draft" if state.get("critiques") or _has_pending_critiques(state) else "synthesize",
        {
            "critique_draft": "critique_draft",
            "synthesize": "synthesize",
        },
    )

    # After critique, always go to synthesize
    graph.add_edge("critique_draft", "synthesize")

    # Synthesize is the end
    graph.add_edge("synthesize", END)

    return graph


def _has_pending_critiques(state: FusionState) -> bool:
    """Check if any critiques were dispatched (indirect check)."""
    # If fan_out_critique_dispatch returned empty list, no critiques were sent
    # In that case, the graph goes straight to synthesize
    return False  # Simplified: rely on conditional edge logic


def compile_fusion_graph():
    """Compile and return the fusion workflow graph."""
    graph = build_fusion_graph()
    return graph.compile()


# ---------------------------------------------------------------------------
# Main entry point — run the fusion workflow
# ---------------------------------------------------------------------------


def run_fusion_blog_generation(
    prompt: str,
    system_prompt: str = "",
    news_headlines: str = "",
) -> dict:
    """Run the multi-LLM fusion workflow to generate a blog post.

    This is the main entry point called by the blog generator pipeline.

    Args:
        prompt: The blog generation prompt with all market data
        system_prompt: System instructions for the LLMs
        news_headlines: Recent news headlines for context

    Returns:
        dict with keys: final_blog, providers_used, errors, duration_seconds, draft_count
    """
    if not settings.has_any_llm():
        logger.error("No LLM providers configured. Cannot generate blog.")
        return {
            "final_blog": "",
            "providers_used": [],
            "errors": ["No LLM providers configured"],
            "duration_seconds": 0,
            "draft_count": 0,
        }

    if not settings.enable_llm:
        logger.info("LLM generation disabled via NTI_ENABLE_LLM=false")
        return {
            "final_blog": "",
            "providers_used": [],
            "errors": ["LLM generation disabled"],
            "duration_seconds": 0,
            "draft_count": 0,
        }

    logger.info("Starting LangGraph fusion blog generation workflow")

    # Compile the graph
    app = compile_fusion_graph()

    # Run the workflow
    initial_state = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "news_headlines": news_headlines,
    }

    # Build Langfuse callback config if enabled
    config = {}
    langfuse_handler = _get_langfuse_handler()
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
        logger.info("Langfuse tracing enabled for fusion workflow")

    try:
        result = app.invoke(initial_state, config=config or None)

        return {
            "final_blog": result.get("final_blog", ""),
            "providers_used": result.get("providers_used", []),
            "errors": result.get("errors", []),
            "duration_seconds": result.get("total_duration_seconds", 0),
            "draft_count": len([d for d in result.get("drafts", []) if d.get("content")]),
        }

    except Exception as e:
        logger.error(f"Fusion workflow failed: {e}")
        return {
            "final_blog": "",
            "providers_used": [],
            "errors": [f"Fusion workflow error: {e}"],
            "duration_seconds": 0,
            "draft_count": 0,
        }


# ---------------------------------------------------------------------------
# Sequential fallback — if LangGraph fails or only one provider
# ---------------------------------------------------------------------------


def run_sequential_generation(prompt: str, system_prompt: str = "") -> dict:
    """Fallback: try LLM providers sequentially until one succeeds.

    Used when fusion mode is "sequential" or as a fallback if LangGraph fails.
    """
    providers = settings.get_enabled_llm_providers()

    # Build Langfuse callback config if enabled
    config = {}
    langfuse_handler = _get_langfuse_handler()
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    for provider in providers:
        try:
            logger.info(f"Sequential mode: trying {provider.name} ({provider.model})")
            llm = _make_llm(provider)

            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            response = llm.invoke(messages, config=config or None)

            return {
                "final_blog": response.content,
                "providers_used": [provider.name],
                "errors": [],
                "duration_seconds": 0,
                "draft_count": 1,
            }

        except Exception as e:
            logger.warning(f"Provider {provider.name} failed: {e}. Trying next...")
            continue

    return {
        "final_blog": "",
        "providers_used": [],
        "errors": ["All LLM providers failed"],
        "duration_seconds": 0,
        "draft_count": 0,
    }


# ---------------------------------------------------------------------------
# Convenience: generate blog with automatic mode selection
# ---------------------------------------------------------------------------


def generate_blog(
    prompt: str,
    system_prompt: str = "",
    news_headlines: str = "",
) -> dict:
    """Generate a blog post using the configured fusion strategy.

    Automatically selects between:
    - "all" mode: LangGraph multi-LLM fusion (fan-out + critique + synthesize)
    - "sequential" mode: Try providers one by one with fallback
    """
    if settings.fusion_mode == "sequential":
        return run_sequential_generation(prompt, system_prompt)

    # Default: fusion mode with LangGraph
    try:
        return run_fusion_blog_generation(prompt, system_prompt, news_headlines)
    except Exception as e:
        logger.warning(f"LangGraph fusion failed, falling back to sequential: {e}")
        return run_sequential_generation(prompt, system_prompt)

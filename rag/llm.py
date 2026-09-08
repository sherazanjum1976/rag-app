"""
llm.py

Wraps the Groq API for LLM inference. Handles prompt construction from
retrieved context (grounded, hallucination-resistant) and robust error
handling around the Groq API call itself.
"""

from typing import List

from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

import config
from rag.vector_store import SearchResult

SYSTEM_PROMPT = """You are a careful, precise document Q&A assistant.

You will be given CONTEXT extracted from a user-uploaded PDF document, \
followed by a QUESTION. Follow these rules strictly:

1. Answer the question using ONLY the information in the provided CONTEXT.
2. If the context does not contain enough information to answer the \
question, clearly say: "I couldn't find this information in the \
document." Do not guess or invent facts.
3. Clearly distinguish between:
   - Information explicitly stated in the document (state it directly).
   - Reasonable inferences drawn from the document (label them as such, \
e.g. "Based on the document, it can be inferred that...").
4. Be concise but complete. Prefer clear prose or short bullet points \
over unnecessary padding.
5. When you use information from the context, mention which page(s) it \
came from, using the page labels given in the context (e.g. "(page 3)").
6. Never fabricate page numbers, quotes, or facts not present in the \
context."""


class LLMError(Exception):
    """Raised when the Groq API call fails or is misconfigured."""


def build_context_block(results: List[SearchResult]) -> str:
    """Format retrieved chunks into a labeled context block for the prompt."""
    if not results:
        return "(No relevant context was found in the document.)"

    blocks = []
    for i, result in enumerate(results, start=1):
        blocks.append(
            f"[Excerpt {i} - {result.chunk.page_label}]\n{result.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: List[SearchResult]) -> str:
    context_block = build_context_block(results)
    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer the question following the system instructions."
    )


class GroqLLM:
    """Thin wrapper around the Groq chat completions API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.model = model or config.LLM_MODEL
        resolved_key = api_key or config.GROQ_API_KEY
        if not resolved_key:
            raise LLMError(
                "No Groq API key configured. Set GROQ_API_KEY in your .env "
                "file (local) or in Streamlit Secrets (cloud deployment)."
            )
        self.client = Groq(api_key=resolved_key)

    def generate_answer(
        self,
        question: str,
        results: List[SearchResult],
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ) -> str:
        """
        Send a RAG prompt (retrieved context + question) to Groq and return
        the generated answer text.
        """
        user_prompt = build_user_prompt(question, results)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except AuthenticationError as exc:
            raise LLMError(
                "Groq authentication failed. Please check that your "
                "GROQ_API_KEY is valid."
            ) from exc
        except RateLimitError as exc:
            raise LLMError(
                "Groq rate limit exceeded. Please wait a moment and try again."
            ) from exc
        except APIConnectionError as exc:
            raise LLMError(
                "Could not connect to the Groq API. Please check your "
                "internet connection and try again."
            ) from exc
        except APIStatusError as exc:
            raise LLMError(
                f"Groq API returned an error (status {exc.status_code}): "
                f"{exc.message}"
            ) from exc
        except APIError as exc:
            raise LLMError(f"Groq API error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Unexpected error calling Groq API: {exc}") from exc

        if not completion.choices:
            raise LLMError("Groq returned an empty response with no choices.")

        answer = completion.choices[0].message.content
        if not answer or not answer.strip():
            raise LLMError("Groq returned an empty answer.")

        return answer.strip()

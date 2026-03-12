"""LLM-based answer generation with source grounding."""

from typing import List

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.models.schemas import DocumentChunk, SourceCitation


SYSTEM_PROMPT = """You are the Mysuru Civic Intelligence System (MCIS), a helpful assistant for civic and municipal matters in Mysuru, Karnataka.

Your role is to answer questions using ONLY the provided context from official Mysuru municipal documents.

IMPORTANT: Always respond in Kannada (ಕನ್ನಡ). Write your entire answer in Kannada.

RULES:
1. Base your answer strictly on the context provided. Do not add information from outside the context.
2. If the context does not contain relevant information, say "ಈ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಲಭ್ಯ ದಾಖಲೆಗಳಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ."
3. Always cite which document(s) you are referring to when giving information.
4. Be concise and factual. Use clear, simple Kannada.
5. For procedural questions (permits, applications, etc.), provide step-by-step guidance when the context supports it."""

USER_PROMPT_TEMPLATE = """Context from Mysuru municipal documents:

{context}

---
Question: {query}

Provide a helpful answer based on the context above in Kannada (ಕನ್ನಡ). Cite the source document(s) in your response."""


class Generator:
    """Generates answers using Ollama 7B instruct model."""

    def __init__(self):
        self.settings = get_settings()

    def _build_context(self, chunks: List[tuple[DocumentChunk, float]]) -> str:
        """Build context string from retrieved chunks with source labels."""
        parts = []
        for i, (chunk, score) in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk.document_name}]\n{chunk.content}"
            )
        return "\n\n".join(parts)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for completion."""
        url = f"{self.settings.llm.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.llm.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.settings.llm.temperature,
                "num_predict": self.settings.llm.max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()
        except httpx.ConnectError as e:
            raise LLMError(
                f"Cannot connect to Ollama at {self.settings.llm.base_url}. "
                "Ensure Ollama is running and the model is pulled (e.g., ollama pull mistral:7b-instruct)."
            ) from e
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama API error: {e}") from e
        except Exception as e:
            raise LLMError(f"Generation failed: {e}") from e

    def generate(
        self,
        query: str,
        chunks: List[tuple[DocumentChunk, float]],
    ) -> str:
        """
        Generate an answer grounded in the provided chunks.
        """
        if not chunks:
            return (
                "ಈ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಲಭ್ಯ ದಾಖಲೆಗಳಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ. "
                "ದಯವಿಟ್ಟು ಪ್ರಶ್ನೆಯನ್ನು ಬೇರೆ ರೀತಿಯಲ್ಲಿ ಕೇಳಿ ಅಥವಾ ದಾಖಲೆಗಳನ್ನು ಸೇರಿಸಲಾಗಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ."
            )

        context = self._build_context(chunks)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)

        # Ollama chat format for instruct models
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

        return self._call_ollama(full_prompt)

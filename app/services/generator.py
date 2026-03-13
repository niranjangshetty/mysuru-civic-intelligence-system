"""LLM-based answer generation with source grounding — Groq API."""

import os
from typing import List

from groq import Groq

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.models.schemas import DocumentChunk, SourceCitation


SYSTEM_PROMPT = """You are the Mysuru Civic Intelligence System (MCIS), a helpful assistant for civic and municipal matters in Mysuru, Karnataka.

Your role is to answer questions using ONLY the provided context from official Mysuru municipal documents.

RESPONSE FORMAT (STRICT):
Your response must always follow this exact structure with line breaks:

**ಕನ್ನಡ:**
[Kannada answer with numbered steps on separate lines]

---

**English:**
[English answer with numbered steps on separate lines]

RULES:
1. Base your answer strictly on the provided context. Do not add information from outside the context.
2. Each numbered step (in both Kannada and English) must be on its own bullet point. Do not merge all steps into a single paragraph.
3. Do NOT include any source citations (no "ಮೂಲ:", no "Source:", no "[Source X]", nothing).
4. The answer must end cleanly after the English section without any extra trailing notes or citations.
5. Keep answers detailed, modular and factual in both Kannada and English."""

USER_PROMPT_TEMPLATE = """Context from Mysuru municipal documents:

{context}

---
Question: {query}

Provide your answer strictly in the required format:

**ಕನ್ನಡ:**
[Full answer in Kannada with numbered steps using ೧. ೨. ೩. where appropriate.]

---

**English:**
[Same answer in English with numbered steps using 1. 2. 3. where appropriate.]

Do not include any source citations (no "ಮೂಲ:", no "Source:", no "[Source X]", nothing).
The answer must end cleanly after the English section, and be concise and factual."""


class Generator:
    """Generates answers using Groq API (drop-in replacement for Ollama)."""

    def __init__(self):
        self.settings = get_settings()
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMError("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=api_key)

    def _build_context(self, chunks: List[tuple[DocumentChunk, float]]) -> str:
        """Build context string from retrieved chunks with source labels."""
        parts = []
        for i, (chunk, score) in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk.document_name}]\n{chunk.content}"
            )
        return "\n\n".join(parts)

    def _call_groq(self, user_prompt: str) -> str:
        """Call Groq API for completion."""
        try:
            response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.settings.llm.temperature,
                max_tokens=self.settings.llm.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LLMError(f"Groq API call failed: {e}") from e

    def generate(
        self,
        query: str,
        chunks: List[tuple[DocumentChunk, float]],
    ) -> str:
        """Generate an answer grounded in the provided chunks."""
        if not chunks:
            return (
                "ಈ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಲಭ್ಯ ದಾಖಲೆಗಳಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ. "
                "ದಯವಿಟ್ಟು ಪ್ರಶ್ನೆಯನ್ನು ಬೇರೆ ರೀತಿಯಲ್ಲಿ ಕೇಳಿ ಅಥವಾ ದಾಖಲೆಗಳನ್ನು ಸೇರಿಸಲಾಗಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ."
            )

        context = self._build_context(chunks)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)

        # Debug logging for LLM input
        print(f"[Generator] chunks_count={len(chunks)}")
        first_chunk_content = chunks[0][0].content if chunks else ""
        print(f"[Generator] first_chunk_preview={first_chunk_content[:200]!r}")
        print(f"[Generator] user_prompt=\n{user_prompt}")

        return self._call_groq(user_prompt)

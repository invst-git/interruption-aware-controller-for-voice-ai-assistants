"""llm.py -- response generation via the Claude API."""
import anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, SYSTEM_PROMPT

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def generate_response(conversation_history: list) -> str:
    """conversation_history: list of {"role": "user"|"assistant", "content": str}"""
    response = _client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=conversation_history,
    )
    return response.content[0].text
from typing import Any
from openai import OpenAI

def build_openai_client(
    api_key: str | None,
    base_url: str | None,
    timeout: float = 120.0,
) -> Any:
    """Build OpenAI client with configurable timeout.

    Args:
        api_key: OpenAI API key (or from env if None)
        base_url: Custom API endpoint (or from env if None)
        timeout: Request timeout in seconds (default 120s for government networks)
    """
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

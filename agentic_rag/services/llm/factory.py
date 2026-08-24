"""LLM Provider Factory — creates provider instances from configuration."""

from agentic_rag.config.settings import LLMProviderConfig, Settings, get_settings
from agentic_rag.services.llm.base import BaseLLMProvider


def _get_openai_provider():
    from agentic_rag.services.llm.openai_provider import OpenAIProvider
    return OpenAIProvider


def _get_claude_provider():
    from agentic_rag.services.llm.claude_provider import ClaudeProvider
    return ClaudeProvider


def _get_local_provider():
    from agentic_rag.services.llm.openai_provider import OpenAIProvider
    return OpenAIProvider


class LLMFactory:
    """Factory for creating LLM provider instances."""

    _registry: dict[str, type] = {
        "openai": _get_openai_provider,
        "claude": _get_claude_provider,
        "local": _get_local_provider,
    }

    _instances: dict[str, BaseLLMProvider] = {}

    @classmethod
    def register(cls, name: str, provider_factory) -> None:
        """Register a new provider type (factory callable)."""
        cls._registry[name] = provider_factory

    @classmethod
    def create(cls, provider_name: str, config: LLMProviderConfig) -> BaseLLMProvider:
        """Create or retrieve a cached LLM provider instance."""
        cache_key = f"{provider_name}:{config.model}"

        if cache_key in cls._instances:
            return cls._instances[cache_key]

        provider_factory = cls._registry.get(provider_name)
        if provider_factory is None:
            raise ValueError(
                f"Unknown LLM provider: {provider_name}. "
                f"Available: {list(cls._registry.keys())}"
            )

        provider_cls = provider_factory()
        instance = provider_cls(
            model_name=config.model,
            api_key=config.api_key,
            api_base=config.api_base,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            vision_model=config.vision_model,
        )
        cls._instances[cache_key] = instance
        return instance

    @classmethod
    def get_default(cls, settings: Settings | None = None) -> BaseLLMProvider:
        """Get the default LLM provider from settings."""
        if settings is None:
            settings = get_settings()

        provider_name = settings.default_provider
        config = settings.llm_providers.get(provider_name)
        if config is None:
            raise ValueError(
                f"Default provider '{provider_name}' not configured. "
                f"Available providers: {list(settings.llm_providers.keys())}"
            )
        return cls.create(provider_name, config)

    @classmethod
    def get_by_model(cls, model_name: str, settings: Settings | None = None) -> BaseLLMProvider:
        """Find a provider that supports the given model."""
        if settings is None:
            settings = get_settings()

        for name, config in settings.llm_providers.items():
            if config.model == model_name or config.vision_model == model_name:
                return cls.create(name, config)

        raise ValueError(f"No provider found for model: {model_name}")

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the provider instance cache."""
        cls._instances.clear()


def get_llm(provider_name: str | None = None) -> BaseLLMProvider:
    """Convenience function to get an LLM provider."""
    settings = get_settings()
    if provider_name:
        config = settings.llm_providers.get(provider_name)
        if config is None:
            raise ValueError(f"Provider not configured: {provider_name}")
        return LLMFactory.create(provider_name, config)
    return LLMFactory.get_default(settings)


def get_embedding_provider() -> BaseLLMProvider:
    """Get an LLM provider configured for embeddings.

    Uses the dedicated embedding config. Falls back to the default LLM provider
    if the embedding provider is the same.
    """
    settings = get_settings()
    emb_config = settings.embedding

    # Case 1: Embedding uses a different API endpoint (self-hosted or separate service)
    if emb_config.api_key or emb_config.api_base:
        # Use a dedicated provider config for embedding
        provider_config = LLMProviderConfig(
            api_key=emb_config.api_key,
            api_base=emb_config.api_base or "",
            model=emb_config.model,
            max_tokens=4096,
            temperature=0.0,
        )
        # Determine provider type from api_base (OpenAI-compatible by default)
        provider_name = emb_config.provider
        return LLMFactory.create(provider_name, provider_config)

    # Case 2: Embedding reuses an existing LLM provider's API key
    provider_config = settings.llm_providers.get(emb_config.provider)
    if provider_config and provider_config.api_key:
        # Create a provider instance for embedding (uses same API key)
        return LLMFactory.create(
            emb_config.provider,
            LLMProviderConfig(
                api_key=provider_config.api_key,
                api_base=provider_config.api_base,
                model=emb_config.model,
                max_tokens=4096,
                temperature=0.0,
            ),
        )

    # Case 3: Fallback to default LLM provider
    return LLMFactory.get_default(settings)


def get_embedding_info() -> tuple[int, str]:
    """Get embedding dimension and model name from config."""
    settings = get_settings()
    return settings.embedding.dim, settings.embedding.model

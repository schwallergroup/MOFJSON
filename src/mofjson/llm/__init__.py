"""LLM integration: profiles, provider clients, prompts, and the repair loop.

Phase 10 / 13.2.1.
"""

from mofjson.llm.anthropic import AnthropicClient, anthropic_tool
from mofjson.llm.openai import (
    EPFL_DEFAULT_BASE_URL,
    ChatCompletionsClient,
    EpflClient,
    LLMResponse,
    LoadDotEnv,
    MakeClient,
    OpenAIClient,
    openai_response_format,
)
from mofjson.llm.profiles import (
    PROFILES,
    CountProfileSchemaTokens,
    GetProfileSchema,
    MergeProfileIntoDocument,
    Profile,
    ProfileModel,
)
from mofjson.llm.prompts import (
    PROMPTS,
    FormatPrimer,
    PromptTemplate,
    RenderPrompt,
    ScanTemplateLeaks,
)
from mofjson.llm.repair import (
    ParseResponse,
    Repair,
    RepairReport,
    ValidateAgainstProfile,
)

__all__ = [
    "EPFL_DEFAULT_BASE_URL",
    "PROFILES",
    "PROMPTS",
    "AnthropicClient",
    "ChatCompletionsClient",
    "CountProfileSchemaTokens",
    "EpflClient",
    "FormatPrimer",
    "GetProfileSchema",
    "LLMResponse",
    "LoadDotEnv",
    "MakeClient",
    "MergeProfileIntoDocument",
    "OpenAIClient",
    "ParseResponse",
    "Profile",
    "ProfileModel",
    "PromptTemplate",
    "RenderPrompt",
    "Repair",
    "RepairReport",
    "ScanTemplateLeaks",
    "ValidateAgainstProfile",
    "anthropic_tool",
    "openai_response_format",
]

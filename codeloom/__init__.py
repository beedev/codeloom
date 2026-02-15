# Patch llama_index to support newer models before any imports
# OpenAI models patch - O-series and GPT-4.1 models
try:
    from llama_index.llms.openai import utils as openai_utils

    # O-series reasoning models (200K context)
    _o_models = {
        'o3': 200000, 'o3-2025-04-16': 200000,
        'o3-mini': 200000, 'o3-mini-2025-01-31': 200000,
        'o3-pro': 200000, 'o3-pro-2025-06-10': 200000,
        'o4-mini': 200000, 'o4-mini-2025-04-16': 200000,
    }

    # GPT-4.1 models (128K context)
    _gpt4_models = {'gpt-4.1': 128000, 'gpt-4.1-mini': 128000, 'gpt-4.1-nano': 128000}

    # Combine all new models
    _new_openai_models = {**_o_models, **_gpt4_models}

    for model, ctx in _new_openai_models.items():
        if hasattr(openai_utils, 'GPT4_MODELS'):
            openai_utils.GPT4_MODELS[model] = ctx
        if hasattr(openai_utils, 'ALL_AVAILABLE_MODELS'):
            openai_utils.ALL_AVAILABLE_MODELS[model] = ctx
        if hasattr(openai_utils, 'CHAT_MODELS'):
            openai_utils.CHAT_MODELS[model] = ctx
        # O-series models are reasoning models
        if model.startswith('o') and hasattr(openai_utils, 'O1_MODELS'):
            openai_utils.O1_MODELS.add(model)
except (ImportError, AttributeError):
    pass

# Gemini models patch - add newer models to llama_index's default list
try:
    import llama_index.llms.gemini.base as gemini_base
    # Add newer Gemini models that llama_index doesn't know about
    _new_gemini_models = (
        "models/gemini-2.0-flash-exp",
        "models/gemini-3-pro-preview",
        "models/gemini-3-pro-image-preview",
        "models/gemini-2.5-pro",
        "models/gemini-2.5-flash",
    )
    # Extend the GEMINI_MODELS tuple
    gemini_base.GEMINI_MODELS = gemini_base.GEMINI_MODELS + _new_gemini_models
except (ImportError, AttributeError):
    pass

from .pipeline import LocalRAGPipeline
from .ollama import run_ollama_server

__all__ = [
    "LocalRAGPipeline",
    "run_ollama_server",
]

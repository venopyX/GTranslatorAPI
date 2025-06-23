"""
GTranslatorAPI - A modern, high-performance translation API using Google Translate.

This FastAPI application provides translation services with caching, rate limiting,
comprehensive error handling, and modern Python best practices.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

import aiohttp
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .cache import CacheManager
from .config import Settings
from .exceptions import TranslationError, UnsupportedLanguageError
from .language_codes import LANGUAGE_CODES
from .models import TranslationRequest, TranslationResponse
from .translator import GoogleTranslator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize settings
settings = Settings()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize cache manager
cache_manager = CacheManager()

# Initialize translator
translator = GoogleTranslator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    # Startup
    logger.info("Starting GTranslatorAPI...")
    await cache_manager.initialize()
    await translator.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down GTranslatorAPI...")
    await cache_manager.close()
    await translator.close()


# Initialize FastAPI app
app = FastAPI(
    title="GTranslatorAPI",
    description="A modern, high-performance translation API using Google Translate",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add middlewares
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(TranslationError)
async def translation_error_handler(request, exc: TranslationError):
    """Handle translation errors."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "Translation service unavailable", "detail": str(exc)}
    )


@app.exception_handler(UnsupportedLanguageError)
async def unsupported_language_error_handler(request, exc: UnsupportedLanguageError):
    """Handle unsupported language errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Unsupported language", "detail": str(exc)}
    )


@app.get("/", summary="API Health Check")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "GTranslatorAPI is running",
        "version": "2.0.0",
        "status": "healthy"
    }


@app.get("/health", summary="Detailed Health Check")
async def health_check():
    """Detailed health check endpoint."""
    try:
        # Test translation service
        test_result = await translator.translate("hello", "es", "en")
        service_status = "healthy" if test_result else "degraded"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        service_status = "unhealthy"
    
    return {
        "status": service_status,
        "timestamp": asyncio.get_event_loop().time(),
        "cache_status": await cache_manager.health_check(),
        "supported_languages": len(LANGUAGE_CODES)
    }


@app.get("/languages", summary="Get Supported Languages")
async def get_supported_languages():
    """Get list of all supported languages."""
    return {
        "languages": LANGUAGE_CODES,
        "count": len(LANGUAGE_CODES)
    }


@app.get("/translate", summary="Translate Text (GET)")
@limiter.limit("100/minute")
async def translate_get(
    request,
    text: str = Query(..., description="Text to translate", min_length=1, max_length=5000),
    target_lang: str = Query(..., description="Target language code", regex=r"^[a-z]{2}(-[A-Z]{2})?$"),
    from_lang: str = Query("auto", alias="from_lang", description="Source language code"),
):
    """
    Translate text using GET method.
    
    - **text**: Text to be translated (1-5000 characters)
    - **target_lang**: Target language code (e.g., 'es', 'fr', 'zh-CN')
    - **from_lang**: Source language code (default: 'auto' for auto-detection)
    """
    translation_request = TranslationRequest(
        text=text,
        target_lang=target_lang,
        from_lang=from_lang
    )
    
    return await _perform_translation(translation_request)


@app.post("/translate", summary="Translate Text (POST)")
@limiter.limit("100/minute")
async def translate_post(request, translation_request: TranslationRequest):
    """
    Translate text using POST method.
    
    Accepts a JSON payload with translation parameters.
    """
    return await _perform_translation(translation_request)


@app.post("/translate/batch", summary="Batch Translate Multiple Texts")
@limiter.limit("50/minute")
async def translate_batch(
    request,
    texts: List[str] = Field(..., description="List of texts to translate", max_items=10),
    target_lang: str = Field(..., description="Target language code"),
    from_lang: str = Field("auto", description="Source language code"),
):
    """
    Translate multiple texts in a single request.
    
    - **texts**: List of texts to translate (max 10 items)
    - **target_lang**: Target language code
    - **from_lang**: Source language code (default: 'auto')
    """
    if not texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No texts provided for translation"
        )
    
    # Validate language codes
    if target_lang not in LANGUAGE_CODES.values():
        raise UnsupportedLanguageError(f"Unsupported target language: {target_lang}")
    
    if from_lang != "auto" and from_lang not in LANGUAGE_CODES.values():
        raise UnsupportedLanguageError(f"Unsupported source language: {from_lang}")
    
    # Perform batch translation
    translations = []
    for text in texts:
        translation_request = TranslationRequest(
            text=text,
            target_lang=target_lang,
            from_lang=from_lang
        )
        result = await _perform_translation(translation_request)
        translations.append(result)
    
    return {"translations": translations, "count": len(translations)}


async def _perform_translation(translation_request: TranslationRequest) -> TranslationResponse:
    """Perform translation with caching and error handling."""
    # Generate cache key
    cache_key = f"translate:{hash(translation_request.text)}:{translation_request.from_lang}:{translation_request.target_lang}"
    
    # Check cache first
    cached_result = await cache_manager.get(cache_key)
    if cached_result:
        logger.info("Cache hit for translation request")
        return TranslationResponse(**cached_result)
    
    try:
        # Perform translation
        translated_text = await translator.translate(
            text=translation_request.text,
            target_lang=translation_request.target_lang,
            from_lang=translation_request.from_lang
        )
        
        # Detect source language if auto
        detected_lang = translation_request.from_lang
        if translation_request.from_lang == "auto":
            detected_lang = await translator.detect_language(translation_request.text)
        
        # Create response
        response = TranslationResponse(
            original_text=translation_request.text,
            translated_text=translated_text,
            source_language=detected_lang,
            target_language=translation_request.target_lang,
            confidence=0.95  # Google Translate typically has high confidence
        )
        
        # Cache the result
        await cache_manager.set(cache_key, response.dict(), ttl=3600)  # Cache for 1 hour
        
        logger.info(f"Translation completed: {translation_request.from_lang} -> {translation_request.target_lang}")
        return response
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise TranslationError(f"Failed to translate text: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )
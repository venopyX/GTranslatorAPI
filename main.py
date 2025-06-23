"""
GTranslatorAPI v2.0 - A modern, high-performance translation API using Google Translate.

This FastAPI application provides translation services with caching, rate limiting,
comprehensive error handling, and modern Python best practices.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import aiohttp
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BaseSettings, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Language codes mapping
LANGUAGE_CODES = {
    "Afan Oromo": "om",
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Chichewa": "ny",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Corsican": "co",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Esperanto": "eo",
    "Estonian": "et",
    "Filipino": "tl",
    "Finnish": "fi",
    "French": "fr",
    "Frisian": "fy",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hausa": "ha",
    "Hawaiian": "haw",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hmong": "hmn",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Igbo": "ig",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Khmer": "km",
    "Kinyarwanda": "rw",
    "Korean": "ko",
    "Kurdish (Kurmanji)": "ku",
    "Kyrgyz": "ky",
    "Lao": "lo",
    "Latin": "la",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Mongolian": "mn",
    "Myanmar (Burmese)": "my",
    "Nepali": "ne",
    "Norwegian": "no",
    "Nyanja (Chichewa)": "ny",
    "Odia (Oriya)": "or",
    "Pashto": "ps",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese (Portugal, Brazil)": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Samoan": "sm",
    "Scots Gaelic": "gd",
    "Serbian": "sr",
    "Sesotho": "st",
    "Shona": "sn",
    "Sindhi": "sd",
    "Sinhala (Sinhalese)": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog (Filipino)": "tl",
    "Tajik": "tg",
    "Tamil": "ta",
    "Tatar": "tt",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Turkmen": "tk",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uyghur": "ug",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Zulu": "zu"
}

# Reverse mapping for language name lookup
LANGUAGE_NAMES = {v: k for k, v in LANGUAGE_CODES.items()}


# Custom Exceptions
class TranslationError(Exception):
    """Raised when translation fails."""
    pass


class UnsupportedLanguageError(Exception):
    """Raised when an unsupported language is requested."""
    pass


# Configuration Settings
class Settings(BaseSettings):
    """Application settings."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS settings
    allowed_origins: List[str] = ["*"]
    
    # Redis cache settings
    redis_enabled: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Rate limiting
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    batch_rate_limit_per_minute: int = int(os.getenv("BATCH_RATE_LIMIT_PER_MINUTE", "50"))
    
    # Translation settings
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "10"))
    translation_timeout: int = int(os.getenv("TRANSLATION_TIMEOUT", "30"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False


# Pydantic Models
class TranslationRequest(BaseModel):
    """Request model for translation."""
    
    text: str = Field(
        ..., 
        description="Text to translate",
        min_length=1,
        max_length=5000,
        example="Hello, how are you?"
    )
    target_lang: str = Field(
        ...,
        description="Target language code",
        example="es"
    )
    from_lang: str = Field(
        "auto",
        description="Source language code (auto for auto-detection)",
        example="en"
    )
    
    @validator("target_lang")
    def validate_target_language(cls, v):
        """Validate target language code."""
        if v not in LANGUAGE_CODES.values():
            raise ValueError(f"Unsupported target language: {v}")
        return v
    
    @validator("from_lang")
    def validate_source_language(cls, v):
        """Validate source language code."""
        if v != "auto" and v not in LANGUAGE_CODES.values():
            raise ValueError(f"Unsupported source language: {v}")
        return v
    
    @validator("text")
    def validate_text(cls, v):
        """Validate text content."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class TranslationResponse(BaseModel):
    """Response model for translation."""
    
    original_text: str = Field(..., description="Original text")
    translated_text: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Detected or specified source language")
    target_language: str = Field(..., description="Target language")
    confidence: Optional[float] = Field(None, description="Translation confidence score")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "original_text": "Hello, how are you?",
                "translated_text": "Hola, ¿cómo estás?",
                "source_language": "en",
                "target_language": "es",
                "confidence": 0.95
            }
        }


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation."""
    
    texts: List[str] = Field(
        ..., 
        description="List of texts to translate", 
        max_items=10,
        min_items=1
    )
    target_lang: str = Field(..., description="Target language code")
    from_lang: str = Field("auto", description="Source language code")
    
    @validator("target_lang")
    def validate_target_language(cls, v):
        """Validate target language code."""
        if v not in LANGUAGE_CODES.values():
            raise ValueError(f"Unsupported target language: {v}")
        return v
    
    @validator("from_lang")
    def validate_source_language(cls, v):
        """Validate source language code."""
        if v != "auto" and v not in LANGUAGE_CODES.values():
            raise ValueError(f"Unsupported source language: {v}")
        return v
    
    @validator("texts")
    def validate_texts(cls, v):
        """Validate texts list."""
        if not v:
            raise ValueError("Texts list cannot be empty")
        
        cleaned_texts = []
        for text in v:
            if not text.strip():
                raise ValueError("Text cannot be empty or whitespace only")
            cleaned_texts.append(text.strip())
        
        return cleaned_texts


# In-memory cache for when Redis is not available
class MemoryCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        if key in self.cache:
            entry = self.cache[key]
            if entry["expires"] > time.time():
                return entry["value"]
            else:
                del self.cache[key]
        
        return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Set cached value with TTL."""
        # Simple LRU eviction
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "value": value,
            "expires": time.time() + ttl
        }
    
    async def health_check(self) -> str:
        """Check cache health."""
        return "memory"


# Cache Manager
class CacheManager:
    """Cache manager with Redis fallback to in-memory."""
    
    def __init__(self):
        self.redis = None
        self.memory_cache = MemoryCache()
        self.enabled = settings.redis_enabled
    
    async def initialize(self):
        """Initialize cache."""
        if not self.enabled:
            logger.info("Cache using in-memory storage")
            return
        
        try:
            import aioredis
            self.redis = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis.ping()
            logger.info("Cache initialized with Redis")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Redis, using in-memory cache: {e}")
            self.redis = None
    
    async def close(self):
        """Close cache connections."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        if self.redis:
            try:
                value = await self.redis.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        return await self.memory_cache.get(key)
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Set cached value with TTL."""
        if self.redis:
            try:
                serialized_value = json.dumps(value, ensure_ascii=False)
                await self.redis.setex(key, ttl, serialized_value)
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        await self.memory_cache.set(key, value, ttl)
    
    async def health_check(self) -> str:
        """Check cache health status."""
        if self.redis:
            try:
                await self.redis.ping()
                return "redis"
            except Exception:
                return "redis_unhealthy"
        
        return await self.memory_cache.health_check()


# Google Translator
class GoogleTranslator:
    """Async Google Translator with connection pooling and retry logic."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = "https://translate.googleapis.com/translate_a/single"
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def initialize(self):
        """Initialize the HTTP session with optimized settings."""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; GTranslatorAPI/2.0)"
            }
        )
        
        logger.info("Google Translator initialized")
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            logger.info("Google Translator closed")
    
    async def translate(
        self,
        text: str,
        target_lang: str,
        from_lang: str = "auto"
    ) -> str:
        """Translate text using Google Translate API."""
        if not self.session:
            raise TranslationError("Translator not initialized")
        
        params = {
            "client": "gtx",
            "sl": from_lang,
            "tl": target_lang,
            "dt": ["t", "bd"],
            "q": text
        }
        
        url = f"{self.base_url}?{urlencode(params, doseq=True)}"
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._extract_translation(data)
                    elif response.status == 429:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise TranslationError(f"HTTP {response.status}: {await response.text()}")
                        
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise TranslationError(f"Network error: {str(e)}")
                
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Network error, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            
            except json.JSONDecodeError as e:
                raise TranslationError(f"Invalid response format: {str(e)}")
        
        raise TranslationError("Max retries exceeded")
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of the given text."""
        if not self.session:
            raise TranslationError("Translator not initialized")
        
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": "en",
            "dt": ["t", "at"],
            "q": text[:100]
        }
        
        url = f"{self.base_url}?{urlencode(params, doseq=True)}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 2 and data[2]:
                        return data[2]
                    return "unknown"
                else:
                    logger.warning(f"Language detection failed: HTTP {response.status}")
                    return "unknown"
                    
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "unknown"
    
    def _extract_translation(self, data) -> str:
        """Extract translated text from Google Translate response."""
        try:
            if not data or not isinstance(data, list) or not data[0]:
                raise TranslationError("Invalid translation response structure")
            
            translations = []
            for item in data[0]:
                if isinstance(item, list) and len(item) > 0 and item[0]:
                    translations.append(str(item[0]))
            
            if not translations:
                raise TranslationError("No translation found in response")
            
            return "".join(translations).strip()
            
        except (IndexError, TypeError, KeyError) as e:
            raise TranslationError(f"Failed to parse translation response: {str(e)}")


# Initialize settings and components
settings = Settings()
limiter = Limiter(key_func=get_remote_address)
cache_manager = CacheManager()
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


# Exception handlers
@app.exception_handler(TranslationError)
async def translation_error_handler(request: Request, exc: TranslationError):
    """Handle translation errors."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "Translation service unavailable", "detail": str(exc)}
    )


@app.exception_handler(UnsupportedLanguageError)
async def unsupported_language_error_handler(request: Request, exc: UnsupportedLanguageError):
    """Handle unsupported language errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Unsupported language", "detail": str(exc)}
    )


# API Routes
@app.get("/", summary="API Health Check")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "GTranslatorAPI is running",
        "version": "2.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health", summary="Detailed Health Check")
async def health_check():
    """Detailed health check endpoint."""
    try:
        test_result = await translator.translate("hello", "es", "en")
        service_status = "healthy" if test_result else "degraded"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        service_status = "unhealthy"
    
    return {
        "status": service_status,
        "timestamp": time.time(),
        "cache_status": await cache_manager.health_check(),
        "supported_languages": len(LANGUAGE_CODES),
        "port": settings.port
    }


@app.get("/languages", summary="Get Supported Languages")
async def get_supported_languages():
    """Get list of all supported languages."""
    return {
        "languages": LANGUAGE_CODES,
        "count": len(LANGUAGE_CODES)
    }


@app.get("/translate", summary="Translate Text (GET)")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def translate_get(
    request: Request,
    text: str = Query(..., description="Text to translate", min_length=1, max_length=5000),
    target_lang: str = Query(..., description="Target language code"),
    from_lang: str = Query("auto", alias="from_lang", description="Source language code"),
):
    """Translate text using GET method."""
    translation_request = TranslationRequest(
        text=text,
        target_lang=target_lang,
        from_lang=from_lang
    )
    
    return await _perform_translation(translation_request)


@app.post("/translate", summary="Translate Text (POST)")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def translate_post(request: Request, translation_request: TranslationRequest):
    """Translate text using POST method."""
    return await _perform_translation(translation_request)


@app.post("/translate/batch", summary="Batch Translate Multiple Texts")
@limiter.limit(f"{settings.batch_rate_limit_per_minute}/minute")
async def translate_batch(request: Request, batch_request: BatchTranslationRequest):
    """Translate multiple texts in a single request."""
    translations = []
    
    for text in batch_request.texts:
        translation_request = TranslationRequest(
            text=text,
            target_lang=batch_request.target_lang,
            from_lang=batch_request.from_lang
        )
        result = await _perform_translation(translation_request)
        translations.append(result)
    
    return {
        "translations": translations, 
        "count": len(translations),
        "batch_id": f"batch_{int(time.time())}"
    }


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
            confidence=0.95
        )
        
        # Cache the result
        await cache_manager.set(cache_key, response.dict(), ttl=3600)
        
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
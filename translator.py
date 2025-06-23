"""
Google Translator implementation with async support and connection pooling.
"""

import asyncio
import json
import logging
from typing import Optional
from urllib.parse import urlencode

import aiohttp

from .exceptions import TranslationError

logger = logging.getLogger(__name__)


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
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
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
        """
        Translate text using Google Translate API.
        
        Args:
            text: Text to translate
            target_lang: Target language code
            from_lang: Source language code (default: auto)
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: If translation fails
        """
        if not self.session:
            raise TranslationError("Translator not initialized")
        
        params = {
            "client": "gtx",
            "sl": from_lang,
            "tl": target_lang,
            "dt": ["t", "bd"],  # Translation and base
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
                        # Rate limited, wait and retry
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
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language code
        """
        if not self.session:
            raise TranslationError("Translator not initialized")
        
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": "en",  # Translate to English for detection
            "dt": ["t", "at"],  # Translation and auto-detect
            "q": text[:100]  # Use first 100 chars for detection
        }
        
        url = f"{self.base_url}?{urlencode(params, doseq=True)}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract detected language from response
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
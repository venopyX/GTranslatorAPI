"""
Custom exceptions for the translation API.
"""


class TranslationError(Exception):
    """Raised when translation fails."""
    pass


class UnsupportedLanguageError(Exception):
    """Raised when an unsupported language is requested."""
    pass


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class CacheError(Exception):
    """Raised when cache operations fail.""" 
    pass
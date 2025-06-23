"""
Pydantic models for request/response validation.
"""

from typing import Optional

from pydantic import BaseModel, Field, validator

from .language_codes import LANGUAGE_CODES


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
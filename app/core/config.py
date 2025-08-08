import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application configuration"""
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    
    # Application Settings
    APP_NAME: str = "Intelligent Query-Retrieval System"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Vector Database Settings
    VECTOR_INDEX_NAME: str = os.getenv("VECTOR_INDEX_NAME", "policy-index")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))  # Gemini default
    
    # Document Processing Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    
    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    DEFAULT_EMBEDDING_PROVIDER: str = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "gemini")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "800"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    
    def validate(self):
        """Validate required environment variables"""
        errors = []
        
        if not self.GEMINI_API_KEY and not self.OPENAI_API_KEY and not self.DEEPSEEK_API_KEY:
            errors.append("At least one LLM API key must be set (GEMINI_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY)")
        
        if errors:
            raise ValueError("Configuration errors: " + ", ".join(errors))


settings = Settings()
# Validate configuration on import
settings.validate()

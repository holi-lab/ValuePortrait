from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class ApiResponse:
    """Standardized response format across different providers"""
    content: str
    raw_response: Any
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    reasoning: Optional[str] = None

class BaseApiClient(ABC):
    """Abstract base class for API clients"""
    @abstractmethod
    def setup_client(self, api_key: str, **kwargs) -> Any:
        """Setup the API client with provider-specific configuration"""
        pass
    
    @abstractmethod
    def make_api_call(self,
                     messages: List[Dict[str, str]],
                     model: str,
                     temperature: float = 0,
                     max_tokens: Optional[int] = None,
                     logger: Optional[logging.Logger] = None,
                     **kwargs) -> ApiResponse:
        """Make API call with standardized parameters"""
        pass
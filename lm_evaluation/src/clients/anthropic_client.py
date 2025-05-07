from anthropic import Anthropic, APIError, APIStatusError, APIConnectionError
from anthropic.types import Message
from .base import BaseApiClient, ApiResponse
import logging
import json
from typing import List, Dict, Optional
import requests

class AnthropicClient(BaseApiClient):
    def setup_client(self, api_key: str, **kwargs) -> Anthropic:
        self.client = Anthropic(api_key=api_key)
        return self.client
    
    def make_api_call(self,
                     messages: List[Dict[str, str]],
                     model: str,
                     temperature: float = 0,
                     max_tokens: Optional[int] = None,
                     logger: Optional[logging.Logger] = None,
                     **kwargs) -> ApiResponse:
        # Convert to Anthropic's messages format
        formatted_messages = []
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Remove seed from kwargs if present
        if 'seed' in kwargs:
            kwargs.pop('seed')

        params = {
            'model': model,
            'messages': formatted_messages,
            'temperature': temperature,
            'max_tokens': max_tokens if max_tokens else 64,
            **kwargs
        }
        
        if logger:
            logger.debug("Anthropic API Request Parameters: %s", 
                        json.dumps({k: v for k, v in params.items() if k != 'messages'}, indent=2))
            logger.debug("Request Messages: %s", 
                        json.dumps([{k: v for k, v in m.items() if k != 'content'} for m in formatted_messages], indent=2))

        try:
            response = self.client.messages.create(**params)
            
            if logger:
                logger.debug("Response received successfully")
                if hasattr(response, 'usage'):
                    logger.debug("Token usage: %s", response.usage)
            
            return ApiResponse(
                content=response.content[0].text,
                raw_response=response,
                usage=getattr(response, 'usage', None),
                model=model
            )

        except APIStatusError as e:
            if logger:
                logger.error("Anthropic API Status Error: %s", str(e))
                
            error_response = requests.Response()
            error_response.status_code = e.status_code
            error_data = {
                "error": {
                    "message": str(e),
                    "code": e.status_code,
                    "type": e.__class__.__name__,
                    "metadata": {
                        "raw": json.dumps(e.response.json() if hasattr(e.response, 'json') else {})
                    }
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.HTTPError(
                f"Anthropic API error: {str(e)}",
                response=error_response
            )
            
        except APIConnectionError as e:
            if logger:
                logger.error("Anthropic API Connection Error: %s", str(e))
            
            error_response = requests.Response()
            error_response.status_code = 503  # Service Unavailable
            error_data = {
                "error": {
                    "message": f"Connection error: {str(e)}",
                    "code": 503,
                    "type": "APIConnectionError"
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.ConnectionError(
                f"Anthropic API connection error: {str(e)}",
                response=error_response
            )
            
        except APIError as e:
            if logger:
                logger.error("Anthropic API General Error: %s", str(e))
            
            error_response = requests.Response()
            error_response.status_code = 500  # Internal Server Error
            error_data = {
                "error": {
                    "message": str(e),
                    "code": 500,
                    "type": "APIError"
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.RequestException(
                f"Anthropic API error: {str(e)}",
                response=error_response
            )
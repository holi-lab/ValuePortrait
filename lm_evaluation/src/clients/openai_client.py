from openai import OpenAI
from openai import OpenAIError, APIError, APIConnectionError, RateLimitError
from .base import BaseApiClient, ApiResponse
import logging
from typing import List, Dict, Optional
import json
import requests

class OpenAIClient(BaseApiClient):
    def setup_client(self, api_key: str, **kwargs) -> OpenAI:
        self.client = OpenAI(api_key=api_key)
        return self.client
    
    def make_api_call(self,
                     messages: List[Dict[str, str]],
                     model: str,
                     temperature: float = 0,
                     max_tokens: Optional[int] = None,
                     logger: Optional[logging.Logger] = None,
                     **kwargs) -> ApiResponse:
        try:
            # Determine which token parameter to use based on model name
            if 'o1' or 'o3' in model:
                params = {
                    'model': model,
                    'messages': messages,
                }
            else:
                # Build parameters dictionary
                params = {
                    'model': model,
                    'messages': messages,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    "seed": kwargs.get('seed'),  # Add seed parameter
                    **kwargs
                }
            
            if logger:
                logger.debug("OpenAI API Request Parameters: %s", 
                            json.dumps({k: v for k, v in params.items() if k != 'messages'}, indent=2))
            
            response = self.client.chat.completions.create(**params)
            
            usage = None
            if hasattr(response, 'usage'):
                usage = {
                    'completion_tokens': response.usage.completion_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                if logger:
                    logger.debug("Token usage: %s", usage)
            
            return ApiResponse(
                content=response.choices[0].message.content,
                raw_response=response,
                usage=usage,
                model=model
            )

        except RateLimitError as e:
            if logger:
                logger.error("OpenAI Rate Limit Error: %s", str(e))
            
            error_response = requests.Response()
            error_response.status_code = 429
            error_data = {
                "error": {
                    "message": str(e),
                    "code": 429,
                    "type": "rate_limit_error"
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.HTTPError(
                f"OpenAI API rate limit error: {str(e)}",
                response=error_response
            )
            
        except APIConnectionError as e:
            if logger:
                logger.error("OpenAI API Connection Error: %s", str(e))
            
            error_response = requests.Response()
            error_response.status_code = 503
            error_data = {
                "error": {
                    "message": f"Connection error: {str(e)}",
                    "code": 503,
                    "type": "api_connection_error"
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.ConnectionError(
                f"OpenAI API connection error: {str(e)}",
                response=error_response
            )
            
        except OpenAIError as e:
            if logger:
                logger.error("OpenAI API Error: %s", str(e))
            
            error_response = requests.Response()
            error_response.status_code = getattr(e, 'status_code', 500)
            error_data = {
                "error": {
                    "message": str(e),
                    "code": getattr(e, 'status_code', 500),
                    "type": e.__class__.__name__
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.RequestException(
                f"OpenAI API error: {str(e)}",
                response=error_response
            )
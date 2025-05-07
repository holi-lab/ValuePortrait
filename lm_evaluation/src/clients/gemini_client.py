import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from .base import BaseApiClient, ApiResponse
import logging
import json
from typing import List, Dict, Optional
import requests

class GeminiClient(BaseApiClient):
    def setup_client(self, api_key: str, **kwargs) -> None:
        genai.configure(api_key=api_key)
        self.client = genai
        return self.client
    
    def make_api_call(self,
                     messages: List[Dict[str, str]],
                     model: str,
                     temperature: float = 0,
                     max_tokens: Optional[int] = None,
                     logger: Optional[logging.Logger] = None,
                     **kwargs) -> ApiResponse:
        try:
            # Convert to Gemini's format (single string with role prefixes)
            formatted_content = []
            for msg in messages:
                prefix = "User: " if msg["role"] == "user" else "Assistant: "
                formatted_content.append(f"{prefix}{msg['content']}")
            
            conversation = "\n".join(formatted_content)
            
            # Extract seed from kwargs if provided
            # seed = kwargs.get('seed')
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens if max_tokens else 64,
            )
            
            if logger:
                logger.debug("Gemini API Request Parameters: %s", 
                            json.dumps({
                                'model': model,
                                'temperature': temperature,
                                'max_tokens': max_tokens,
                            }, indent=2))
                logger.debug("Request Content: %s", conversation)
            
            model_client = self.client.GenerativeModel(model_name=model)
            response = model_client.generate_content(
                conversation,
                generation_config=generation_config
            )
            
            if logger:
                logger.debug("Response received successfully")
                logger.debug(response)

            return ApiResponse(
                content=response.text,
                raw_response=response,
                usage=None,  # Gemini doesn't provide token usage
                model=model
            )

        except google_exceptions.PermissionDenied as e:
            if logger:
                logger.error("Gemini API Permission Error: %s", str(e))
            
            error_response = requests.Response()
            error_response.status_code = 403
            error_data = {
                "error": {
                    "message": str(e),
                    "code": 403,
                    "type": "PermissionDenied",
                    "metadata": {
                        "raw": str(e)
                    }
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.HTTPError(
                f"Gemini API permission error: {str(e)}",
                response=error_response
            )
            
        except google_exceptions.InvalidArgument as e:
            if logger:
                logger.error("Gemini API Invalid Argument Error: %s", str(e))
            
            error_response = requests.Response()
            error_response.status_code = 400
            error_data = {
                "error": {
                    "message": str(e),
                    "code": 400,
                    "type": "InvalidArgument",
                    "metadata": {
                        "raw": str(e)
                    }
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.HTTPError(
                f"Gemini API invalid argument error: {str(e)}",
                response=error_response
            )
            
        except google_exceptions.ResourceExhausted as e:
            if logger:
                logger.error("Gemini API Rate Limit Error: %s", str(e))
            
            error_response = requests.Response()
            error_response.status_code = 429
            error_data = {
                "error": {
                    "message": str(e),
                    "code": 429,
                    "type": "ResourceExhausted",
                    "metadata": {
                        "raw": str(e)
                    }
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.HTTPError(
                f"Gemini API rate limit error: {str(e)}",
                response=error_response
            )
            
        except google_exceptions.ServiceUnavailable as e:
            if logger:
                logger.error("Gemini API Service Unavailable Error: %s", str(e))
            
            error_response = requests.Response()
            error_response.status_code = 503
            error_data = {
                "error": {
                    "message": str(e),
                    "code": 503,
                    "type": "ServiceUnavailable",
                    "metadata": {
                        "raw": str(e)
                    }
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.ConnectionError(
                f"Gemini API service unavailable: {str(e)}",
                response=error_response
            )
            
        except Exception as e:
            if logger:
                logger.error("Gemini API General Error: %s", str(e))
            
            error_response = requests.Response()
            error_response.status_code = 500
            error_data = {
                "error": {
                    "message": str(e),
                    "code": 500,
                    "type": "GeneralError",
                    "metadata": {
                        "raw": str(e)
                    }
                }
            }
            error_response._content = json.dumps(error_data).encode()
            
            raise requests.exceptions.RequestException(
                f"Gemini API error: {str(e)}",
                response=error_response
            )
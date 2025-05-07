import requests
from typing import List, Dict, Optional
import logging
from .base import BaseApiClient, ApiResponse
import json

def get_provider_order(model: str) -> list[str]:
    """
    Determines the provider order based on the model name.
    
    Args:
        model: The name of the model (e.g., 'google/gemini-pro-1.5')
        
    Returns:
        A list containing the appropriate provider(s)
    """
    # Google AI Studio models
    if model.startswith(('google/gemini-')):
        return ["Google AI Studio"]
    
    # DeepSeek models
    if model.startswith('deepseek/deepseek-'):
        return ["DeepInfra"]
    
    # Alibaba (Qwen) models
    if model.startswith('qwen/qwen-'):
        return ["Alibaba"]
    
    # xAI models
    if model.startswith('x-ai/grok-'):
        return ["xAI"]
    
    # Lambda (Meta-Llama) models
    if model.startswith('meta-llama/llama-'):
        return ["Lambda"]
    
    # Mistral models
    if model.startswith('mistralai/mistral-'):
        return ["Mistral"]
    
    # QwQ models
    if model.startswith('qwen/qwq-32b'):
        return ["DeepInfra"]
    
    # Anthropic models
    if model.startswith(('anthropic')):
        return ["Anthropic"]
    
    # Default case - return empty list or raise error depending on requirements
    raise ValueError(f"Unknown model: {model}")

class OpenRouterClient(BaseApiClient):
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def setup_client(self, api_key: str, site_url: Optional[str] = None, site_name: Optional[str] = None, **kwargs) -> None:
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        if site_url:
            self.headers["HTTP-Referer"] = site_url
        if site_name:
            self.headers["X-Title"] = site_name
            
        return self
    
    def make_api_call(self,
                     messages: List[Dict[str, str]],
                     model: str,
                     temperature: float = 0,
                     max_tokens: Optional[int] = None,
                     logger: Optional[logging.Logger] = None,
                     **kwargs) -> ApiResponse:
        url = f"{self.BASE_URL}/chat/completions"
        
        try:
            provider_order = get_provider_order(model)
        except ValueError as e:
            if logger:
                logger.warning(f"Provider order not found for model {model}: {str(e)}")
            provider_order = []  # Or handle this case according to your requirements
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": kwargs.get('seed'),
            "provider": {
                "order": provider_order,
                "allow_fallbacks": False,
                "require_parameters": True,
                # "data_collection": "deny"
            }
        }
        
        for k, v in kwargs.items():
            if v is not None and k != 'seed':
                params[k] = v

        if model.startswith('anthropic'):
            params.pop("seed", None)
                
        if max_tokens is not None and model != "deepseek/deepseek-r1":
            params["max_tokens"] = max_tokens
            
        if logger:
            logger.debug("OpenRouter API Request: %s", url)
            logger.debug("Request Headers: %s", {k: v for k, v in self.headers.items() if k != 'Authorization'})
            logger.debug("Request Parameters: %s", json.dumps(params, indent=2))
        
        try:
            response = requests.post(
                url=url,
                headers=self.headers,
                json=params
            )
            
            try:
                response_data = response.json()
                if logger:
                    logger.debug("Response: [%d] %s", response.status_code, json.dumps(response_data, indent=2))
                
                if 'error' in response_data:
                    error_msg = response_data['error'].get('message', 'Unknown error')
                    error_code = response_data['error'].get('code')
                    
                    if 'metadata' in response_data['error']:
                        try:
                            raw_error = json.loads(response_data['error']['metadata']['raw'])
                            if 'error' in raw_error:
                                error_msg = f"{error_msg} - {raw_error['error'].get('message', '')}"
                        except (json.JSONDecodeError, KeyError):
                            pass
                    
                    error_response = requests.Response()
                    error_response.status_code = error_code
                    error_response._content = json.dumps(response_data).encode()
                    error_response.headers = response.headers
                    
                    raise requests.exceptions.HTTPError(
                        f"OpenRouter API error: {error_msg}",
                        response=error_response
                    )
                
                response.raise_for_status()
                
                if logger:
                    if usage := response_data.get("usage"):
                        logger.debug("Token usage: %s", usage)
                    if provider := response_data.get("provider"):
                        logger.debug("Provider used: %s", provider)
                message = response_data["choices"][0]["message"]
                reasoning = message.get("reasoning")

                return ApiResponse(
                    content=message["content"],
                    raw_response=response_data,
                    usage=response_data.get("usage"),
                    model=model,
                    reasoning=reasoning if reasoning is not None else None
                )
                
            except json.JSONDecodeError as e:
                if logger:
                    logger.error("Failed to parse response as JSON: %s", response.text)
                raise
                
        except requests.exceptions.RequestException as e:
            if logger:
                logger.error("OpenRouter API Error: %s", str(e))
            raise
import random
import time
import logging
import json
from typing import Any, Callable, Optional
import requests
from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError

class RetryHandler:
    def __init__(self,
                max_retries: int = 5,
                base_delay: float = 1.0,
                max_delay: float = 60.0,
                exponential_base: float = 2.0,
                jitter: bool = True,
                logger: Optional[logging.Logger] = None):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.logger = logger

    def calculate_delay(self, attempt: int, retry_after: Optional[str] = None) -> float:
        # If we have a Retry-After header, use that as the base delay
        if retry_after:
            try:
                delay = float(retry_after)
            except (ValueError, TypeError):
                delay = self.base_delay * (self.exponential_base ** attempt)
        else:
            delay = self.base_delay * (self.exponential_base ** attempt)

        delay = min(delay, self.max_delay)

        if self.jitter:
            jitter_amount = delay * 0.1
            delay += random.uniform(-jitter_amount, jitter_amount)

        return delay

    def _extract_status_code(self, error: Exception) -> Optional[int]:
        """Extract status code from various error response formats."""
        if isinstance(error, requests.exceptions.RequestException):
            if hasattr(error, 'response') and error.response is not None:
                # Try to get direct status code from response
                if error.response.status_code:
                    return error.response.status_code
                
                # If no status code, try to parse JSON response for nested error codes
                try:
                    response_json = error.response.json()
                    # Check for Google AI Studio style error
                    if isinstance(response_json, dict):
                        if 'error' in response_json:
                            if isinstance(response_json['error'], dict):
                                if 'code' in response_json['error']:
                                    return response_json['error']['code']
                                # Check for nested metadata
                                if 'metadata' in response_json['error']:
                                    try:
                                        raw_error = json.loads(response_json['error']['metadata']['raw'])
                                        if 'error' in raw_error and 'code' in raw_error['error']:
                                            return raw_error['error']['code']
                                    except (json.JSONDecodeError, KeyError, TypeError):
                                        pass
                except (ValueError, AttributeError):
                    pass
        return None

    def should_retry(self, error: Exception) -> bool:
        # Handle OpenRouter/HTTP specific errors
        if isinstance(error, requests.exceptions.RequestException):
            # Extract status code using the helper method
            status_code = self._extract_status_code(error)
            if status_code is not None and status_code in [408, 429, 500, 502, 503, 504]:
                return True
                
            # Retry on connection errors
            if isinstance(error, (requests.exceptions.ConnectionError,
                                requests.exceptions.Timeout)):
                return True
            
            # Check for quota exhaustion in response body
            if hasattr(error, 'response') and error.response is not None:
                try:
                    response_json = error.response.json()
                    if 'error' in response_json:
                        error_msg = str(response_json['error'].get('message', '')).lower()
                        if 'quota' in error_msg or 'resource exhausted' in error_msg:
                            return True
                except (ValueError, AttributeError):
                    pass
            return False

        # Handle OpenAI-style errors
        retryable_errors = (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            APIError
        )
        return isinstance(error, retryable_errors)

    def get_retry_after(self, error: Exception) -> Optional[str]:
        """Extract Retry-After header from error response if available."""
        if isinstance(error, requests.exceptions.RequestException):
            if hasattr(error, 'response') and error.response is not None:
                return error.response.headers.get('Retry-After')
        return None

    def execute_with_retry(self,
                         func: Callable[..., Any],
                         *args: Any,
                         **kwargs: Any) -> Any:
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == self.max_retries or not self.should_retry(e):
                    if self.logger:
                        self.logger.error(
                            f"Final retry attempt failed or non-retryable error: {str(e)}",
                            exc_info=True
                        )
                    raise

                # Get Retry-After header if available
                retry_after = self.get_retry_after(e)
                delay = self.calculate_delay(attempt, retry_after)

                if self.logger:
                    error_details = str(e)
                    if isinstance(e, requests.exceptions.RequestException) and hasattr(e, 'response'):
                        try:
                            error_details = e.response.json()
                        except (ValueError, AttributeError):
                            error_details = e.response.text if e.response else str(e)

                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {error_details}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                time.sleep(delay)

        raise last_exception
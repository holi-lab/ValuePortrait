from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import os
import multiprocessing
from tqdm import tqdm
from ..clients.openai_client import OpenAIClient
from ..clients.gemini_client import GeminiClient
from ..clients.anthropic_client import AnthropicClient
from ..clients.openrouter_client import OpenRouterClient
from ..utils.retry import RetryHandler
from ..utils.config import save_json
from .response_parser import parse_likert_response, map_response_to_numeric
from .experiment import ExperimentConfig
from ..clients.factory import ApiClientFactory
from .prompt_loader import load_prompt_templates
from ..utils.logging import setup_logger, get_process_logger
import time

# def has_correlations(output: Dict[str, Any]) -> bool:
#     """Check if output has correlations or bfi_correlations"""
#     return 'correlations' in output or 'bfi_correlations' in output

def get_prompt_template(portrait_id: int, prompt_templates: Dict[str, str]) -> tuple[str, str]:
    """Get the appropriate prompt template based on portrait_id"""
    first_digit = int(str(portrait_id)[0])
    if first_digit in [1, 2]:
        return prompt_templates['reddit'], 'reddit'
    elif first_digit in [3, 4]:
        return prompt_templates['sharegpt'], 'sharegpt'
    else:
        raise ValueError(f"Unexpected portrait_id prefix: {first_digit}")

def create_prompt(prompt_template: str, template_type: str, entry: Dict[str, Any], output_content: str) -> str:
    """Create prompt based on template type and entry data"""
    text = entry['content']['text']
    
    if template_type == 'reddit':
        title = entry['content']['title']
        prompt = prompt_template.replace('{title}', title)
        prompt = prompt.replace('{text}', text)
        prompt = prompt.replace('{content}', output_content)
    else:  # sharegpt and lmsys
        prompt = prompt_template.replace('{text}', text)
        prompt = prompt.replace('{content}', output_content)
    
    return prompt

def process_entry(client: Union[OpenAIClient, GeminiClient, AnthropicClient, OpenRouterClient],
                 entry: Dict[str, Any],
                 prompt_templates: Dict[str, str],
                 model: str,
                 logger: logging.Logger) -> List[Dict[str, Any]]:
    """Process a single entry using the prompt template and Llama API"""
    portrait_id = entry['portrait_id']
    results = []
    
    logger.info(f"Processing portrait_id: {portrait_id}")
    
    retry_handler = RetryHandler(
        max_retries=2,
        base_delay=3.0,
        max_delay=5.0,
        jitter=True,
        logger=logger
    )
    
    try:
        prompt_template, template_type = get_prompt_template(portrait_id, prompt_templates)
        logger.debug(f"Using {template_type} template for portrait_id {portrait_id}")
    except ValueError as e:
        logger.error(f"Error with portrait_id {portrait_id}: {str(e)}")
        return [{'portrait_id': portrait_id, 'error': str(e)}]

    for output in entry['outputs']:
        try:
            logger.debug(f"Processing output_id: {output['id']}")
            prompt = create_prompt(prompt_template, template_type, entry, output['content'])
            
            # sleep 6 seconds for avoiding RPM (only for gemini)
            # logger.debug(f"API call completed, sleeping for 6 seconds to avoid rate limits...")
            # time.sleep(6)

            response = retry_handler.execute_with_retry(
                func=client.make_api_call,
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
                max_tokens=64, ## for generic models
                # max_tokens=2048, ## should be longer with reasoning models
                logger=logger,  # Pass the logger to the API call
                seed=42  # Add seed parameter
            )
            # Log the API response details
            logger.debug(f"API response received for portrait_id {portrait_id}, output_id {output['id']}")
            if response.usage:
                logger.debug(f"Usage stats: {response.usage}")

            raw_response = response.content
            logger.debug(f"Raw response: {raw_response}")
            
            parsed_response = parse_likert_response(raw_response)
            logger.debug(f"Parsed response: {parsed_response}")
            
            numeric_response = map_response_to_numeric(parsed_response)
            logger.debug(f"Numeric response: {numeric_response}")

            reasoning_response = ""
            if response.reasoning:
                reasoning_response = response.reasoning
            
            result = {
                'portrait_id': portrait_id,
                'option_id': output['id'],
                'raw_response': raw_response,
                'parsed_response': parsed_response,
                'numeric_response': numeric_response,
                'content': {
                    'title': entry['content'].get('title', ''),
                    'text': entry['content'].get('text', ''),
                    'output_text': output['content']
                },
                'prompt': prompt,
                'reasoning': reasoning_response,
            }
            
            if 'correlations' in output:
                result['correlations'] = output['correlations']
            if 'bfi_correlations' in output:
                result['bfi_correlations'] = output['bfi_correlations']
            if 'higher_pvq_correlations' in output:
                result['higher_pvq_correlations'] = output['higher_pvq_correlations']
            
        except Exception as e:
            logger.error(f"Error processing output {output.get('id')}: {str(e)}")
            result = {
                'portrait_id': portrait_id,
                'option_id': output.get('id'),
                'error': str(e),
                'content': {
                    'title': entry['content'].get('title', ''),
                    'text': entry['content'].get('text', ''),
                    'output_text': output.get('content', '')
                },
                'prompt': prompt,
            }
        
        results.append(result)
    
    return results

def process_batch(args: tuple) -> List[Dict]:
    """Process a batch of entries"""
    batch, provider, model, prompt_templates, api_key = args
    
    # Get logger for this process
    process_logger = get_process_logger()
    
    # Rest of your code remains the same
    client = ApiClientFactory.create_client(provider)
    client.setup_client(api_key)
    results = []
    
    for entry in batch:
        try:
            entry_results = process_entry(client, entry, prompt_templates, model, process_logger)
            results.extend(entry_results)
        except Exception as e:
            process_logger.error(f"Error processing entry {entry.get('portrait_id')}: {str(e)}")
    
    return results

def run_experiment_parallel(
    experiment: ExperimentConfig,
    data: List[Dict],
    base_output_dir: str,
    prompt_dir: str,
    logger: logging.Logger,
    num_processes: int = None
) -> None:
    """Run a single experiment with parallel processing across multiple providers."""
    # Initialize logger with experiment name
    logger = setup_logger(log_dir="logs", experiment_name=experiment.name)

    # Set number of processes to 75% of available CPU cores
    if num_processes is None:
        num_processes = max(1, int(multiprocessing.cpu_count() * 0.75))
    num_processes = max(1, min(num_processes, len(data)))
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create experiment output directory
    exp_output_dir = os.path.join(base_output_dir, experiment.name, timestamp)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    logger.info(f"\n=== Starting Experiment: {experiment.name} with {num_processes} processes ===")
    if experiment.description:
        logger.info(f"Description: {experiment.description}")
    
    # Run for each provider
    for provider, provider_config in experiment.providers.items():
        # Get provider-specific API key
        api_key = os.getenv(f'{provider.upper()}_API_KEY')
        if not api_key:
            logger.error(f"Missing API key for provider {provider}, skipping...")
            continue
        
        # Create provider-specific output directory
        provider_output_dir = os.path.join(exp_output_dir, provider)
        os.makedirs(provider_output_dir, exist_ok=True)
        
        # Run each combination of model and prompt version
        for model in provider_config.models:
            for prompt_version in experiment.prompts:
                try:
                    logger.info(f"\nRunning combination: provider={provider}, model={model}, prompt={prompt_version}")
                    
                    # Load prompt templates for this version
                    try:
                        prompt_templates = load_prompt_templates(prompt_version, prompt_dir, logger)
                    except Exception as e:
                        logger.error(f"Error loading prompt templates for version {prompt_version}: {str(e)}")
                        continue
                    
                    # Calculate total outputs for progress tracking
                    total_outputs = sum(
                        len(entry['outputs'])
                        for entry in data
                    )
                    
                    # Create batches of data
                    batch_size = max(1, len(data) // (num_processes * 4))
                    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
                    
                    logger.info(f"Processing {len(batches)} batches with {num_processes} processes")
                    
                    # Prepare arguments for each batch
                    batch_args = [
                        (batch, provider, model, prompt_templates, api_key)
                        for batch in batches
                    ]
                    
                    # Initialize progress bar
                    pbar = tqdm(total=total_outputs, 
                              desc=f"Processing {provider}-{model}-{prompt_version}",
                              unit="output")
                    
                    # Initialize counters for tracking progress
                    processed_count = multiprocessing.Value('i', 0)
                    errors_count = multiprocessing.Value('i', 0)
                    
                    # Process batches in parallel
                    with multiprocessing.Pool(num_processes) as pool:
                        all_results = []
                        for batch_results in pool.imap_unordered(process_batch, batch_args):
                            all_results.extend(batch_results)
                            
                            # Update counts and progress bar
                            successful = sum(1 for r in batch_results if 'error' not in r)
                            errors = sum(1 for r in batch_results if 'error' in r)
                            
                            with processed_count.get_lock():
                                processed_count.value += successful
                            with errors_count.get_lock():
                                errors_count.value += errors
                            
                            pbar.update(successful + errors)
                    
                    pbar.close()
                    
                    # Save results for this combination
                    model_name = model.split('/')[-1]
                    output_filename = f"{model_name}_{prompt_version}_results.json"
                    output_path = os.path.join(provider_output_dir, output_filename)
                    
                    try:
                        save_json(all_results, output_path, logger)
                    except Exception as e:
                        logger.error(f"Error saving results to {output_path}: {str(e)}")
                        continue
                    
                    logger.info(f"\n--- Run Summary for {provider}-{model}-{prompt_version} ---")
                    logger.info(f"Total processed: {processed_count.value + errors_count.value}")
                    logger.info(f"Successful: {processed_count.value}")
                    logger.info(f"Errors: {errors_count.value}")
                    logger.info(f"Results saved to: {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error in combination {provider}-{model}-{prompt_version}: {str(e)}")
                    continue
        
        logger.info(f"\n=== Completed Provider: {provider} ===")
    
    logger.info(f"\n=== Completed Experiment: {experiment.name} ===")
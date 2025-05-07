import os
import multiprocessing
from dotenv import load_dotenv
import sys
from typing import Dict, Any

from src.utils.logging import setup_logger
from src.utils.config import load_config, read_json
from src.core.experiment import ExperimentConfig, ProviderConfig
from src.core.processor import run_experiment_parallel

def create_experiment_config(exp_config: Dict[str, Any]) -> ExperimentConfig:
    """Create ExperimentConfig from dictionary configuration"""
    provider_configs = {}
    for provider, config in exp_config['providers'].items():
        provider_configs[provider] = ProviderConfig(
            models=config['models']
        )
    
    return ExperimentConfig(
        name=exp_config['name'],
        providers=provider_configs,
        prompts=exp_config['prompts'],
        description=exp_config.get('description', '')
    )

def main():
    # Constants
    CONFIG_PATH = 'config/full_config.yaml'
    INPUT_PATH = 'data/Phase1_total.json'
    PROMPT_DIR = 'prompts/'
    OUTPUT_DIR = 'outputs/'
    NUM_PROCESSES = None  # Will use CPU count - 1
    
    # Set up logger
    logger = setup_logger()
    logger.info("=== Starting Processing ===")
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        config = load_config(CONFIG_PATH, logger)
        
        # Load input data once
        data = read_json(INPUT_PATH, logger)
        logger.info(f"Loaded {len(data)} entries from input file")
        
        # Get experiments to run
        experiment_name = sys.argv[1] if len(sys.argv) > 1 else None
        experiments = []
        
        if experiment_name:
            exp_config = next(
                (exp for exp in config['experiments'] if exp['name'] == experiment_name),
                None
            )
            if not exp_config:
                raise ValueError(f"No experiment found with name: {experiment_name}")
            experiments = [exp_config]
        else:
            experiments = config['experiments']
        
        # Run each experiment
        for exp_config in experiments:
            experiment = create_experiment_config(exp_config)
            
            run_experiment_parallel(
                experiment=experiment,
                data=data,
                base_output_dir=OUTPUT_DIR,
                prompt_dir=PROMPT_DIR,
                logger=logger,
                num_processes=NUM_PROCESSES
            )
        
        logger.info("\n=== All Experiments Completed ===")
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
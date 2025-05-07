import os
import yaml
import json
from typing import Dict, List, Any
import logging

def load_config(config_path: str, logger: logging.Logger) -> Dict:
    """Load experiment configuration from YAML file"""
    try:
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.debug("Successfully loaded configuration")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def read_json(file_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Read JSON data from a file"""
    try:
        logger.info(f"Reading JSON file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.debug(f"Successfully read JSON file with {len(data)} entries")
        return data
    except Exception as e:
        logger.error(f"Error reading JSON file: {str(e)}")
        raise

def save_json(data: List[Dict[str, Any]], file_path: str, logger: logging.Logger) -> None:
    """Save data to a JSON file"""
    try:
        logger.info(f"Saving results to: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        logger.debug("Successfully saved results")
    except Exception as e:
        logger.error(f"Error saving JSON file: {str(e)}")
        raise
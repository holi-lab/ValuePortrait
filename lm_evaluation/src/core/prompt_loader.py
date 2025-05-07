import os
from typing import Dict
import logging

def load_prompt_templates(prompt_version: str, prompt_dir: str, logger: logging.Logger) -> Dict[str, str]:
    """Read prompt templates for a specific version"""
    try:
        version_dir = os.path.join(prompt_dir, prompt_version)
        reddit_path = os.path.join(version_dir, 'reddit_prompt.txt')
        sharegpt_path = os.path.join(version_dir, 'sharegpt_prompt.txt')
        
        with open(reddit_path, 'r', encoding='utf-8') as file:
            reddit_prompt = file.read()
        with open(sharegpt_path, 'r', encoding='utf-8') as file:
            sharegpt_prompt = file.read()
        
        templates = {
            'reddit': reddit_prompt,
            'sharegpt': sharegpt_prompt
        }
        logger.debug(f"Successfully loaded prompts for version {prompt_version}")
        return templates
    except Exception as e:
        logger.error(f"Error reading prompt templates: {str(e)}")
        raise
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ProviderConfig:
    """Configuration for a specific provider"""
    models: List[str]

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    name: str
    providers: Dict[str, ProviderConfig]
    prompts: List[str]
    description: str = ""
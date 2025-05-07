import json
import os
from collections import defaultdict
from typing import Dict, List
from copy import deepcopy

def read_json_file(file_path: str) -> Dict:
    """
    Read and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data as a dictionary
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def initialize_result_template(sample_data: Dict) -> Dict:
    """
    Initialize the result dictionary with the same structure as input data.
    
    Args:
        sample_data: A sample of the input JSON data
        
    Returns:
        Template dictionary with the same structure
    """
    template = {
        "meta": {
            "model_name": "averaged_results",
            "prompt": "averaged_centered_means"
        },
        "scores": {}
    }
    
    # Copy the structure of scores
    for category, dimensions in sample_data["scores"].items():
        template["scores"][category] = {
            dim: {
                "centered_mean": 0.0,
                "original_mean": 0.0,
                "std_dev": 0.0,
                "n": 0
            } for dim in dimensions.keys()
        }
    
    return template

def calculate_average_centered_means(file_paths: List[str]) -> Dict:
    """
    Calculate average centered means across multiple JSON files while maintaining
    the original JSON structure.
    
    Args:
        file_paths: List of paths to JSON files
        
    Returns:
        Dictionary containing averaged data in the original format
    """
    if not file_paths:
        raise ValueError("No JSON files provided")
    
    # Initialize using the structure from the first file
    first_data = read_json_file(file_paths[0])
    result = initialize_result_template(first_data)
    
    # Initialize accumulators
    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts = defaultdict(lambda: defaultdict(int))
    
    # Process each file
    for file_path in file_paths:
        data = read_json_file(file_path)
        
        for category, dimensions in data["scores"].items():
            for dimension, values in dimensions.items():
                sums[category][dimension]["centered_mean"] += values["centered_mean"]
                sums[category][dimension]["original_mean"] += values["original_mean"]
                sums[category][dimension]["std_dev"] += values["std_dev"]
                sums[category][dimension]["n"] += values["n"]
                counts[category][dimension] += 1
    
    # Calculate averages
    for category, dimensions in result["scores"].items():
        for dimension in dimensions:
            count = counts[category][dimension]
            if count > 0:
                result["scores"][category][dimension] = {
                    "centered_mean": sums[category][dimension]["centered_mean"] / count,
                    "original_mean": sums[category][dimension]["original_mean"] / count,
                    "std_dev": sums[category][dimension]["std_dev"] / count,
                    "n": int(sums[category][dimension]["n"] / count)
                }
    
    return result

def save_results(results: Dict, output_path: str):
    """
    Save the averaged results to a JSON file.
    
    Args:
        results: Dictionary containing the averaged data
        output_path: Path where the results should be saved
    """
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=2)

def main():
    # Directory containing JSON files
    input_directory = 'gpt_results_wo_reasoning'
    output_file = 'averaged_gpt_results_wo_reasoning_final.json'
    
    # Get all JSON files in the directory
    json_files = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith('.json')
    ]
    
    # Calculate average centered means
    averaged_results = calculate_average_centered_means(json_files)
    
    # Save results
    save_results(averaged_results, output_file)
    
    print(f"Results have been saved to {output_file}")

if __name__ == "__main__":
    main()
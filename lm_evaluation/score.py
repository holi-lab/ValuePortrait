import json
import os
from typing import Dict, List, Union
import math
import logging
import datetime

# Create logs directory if it doesn't exist
log_dir = os.path.join('logs', 'score')
os.makedirs(log_dir, exist_ok=True)

# Create log filename with timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(log_dir, f'dimension_scorer_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class DimensionScorer:
    def __init__(self, input_file: str, output_file: str, threshold: float = 0.3, cor_mode: str = 'abs', center: bool = False):
        self.input_file = input_file
        self.output_file = output_file
        self.threshold = threshold
        self.cor_mode = cor_mode
        self.center = center
        logging.info(f"\nInitializing DimensionScorer:")
        logging.info(f"Input file: {input_file}")
        logging.info(f"Threshold: {threshold}")
        logging.info(f"Correlation mode: {cor_mode}")
        logging.info(f"Centering enabled: {center}")
        self.data = self._load_data()
        self.model_name = self._get_model_name()
        self.prompt = self._get_prompt()
        
    def _load_data(self) -> List[dict]:
        with open(self.input_file, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} entries from {self.input_file}")
            return data
    
    def _get_model_name(self) -> str:
        filename = os.path.basename(self.input_file)
        model_name = os.path.splitext(filename)[0].split('_')[0]
        logging.info(f"Extracted model name: {model_name}")
        return model_name
    
    def _get_prompt(self) -> str:
        filename = os.path.basename(self.input_file)
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        prompt = ""
        if len(parts) == 3:
            prompt = parts[1]
        if len(parts) == 4:
            prompt = parts[1] + "_" + parts[2]
        logging.info(f"Extracted prompt: {prompt}")
        return prompt

    def _check_correlation(self, correlation: float) -> bool:
        if self.cor_mode == 'abs':
            return abs(correlation) > self.threshold
        elif self.cor_mode == 'pos':
            return correlation > self.threshold
        elif self.cor_mode == 'neg':
            return correlation < -self.threshold
        return False

    def calculate_dimension_scores(self):
        logging.info("\nStarting dimension score calculations...")
        
        # Initialize dictionaries to store responses that pass correlation check
        valid_responses = {
            'pvq': [],     # For Schwartz values
            'bfi': [],     # For BFI dimensions
            'higher_pvq': []  # For Higher PVQ dimensions
        }
        
        # Initialize dictionaries to store dimension-specific scores
        dimension_scores = {
            'pvq': {},
            'bfi': {},
            'higher_pvq': {}
        }

        # Process each entry
        for idx, entry in enumerate(self.data):
            numeric_response = entry.get('numeric_response')
            if numeric_response is None:
                continue

            logging.debug(f"\nProcessing entry {idx + 1}:")
            logging.debug(f"Numeric response: {numeric_response}")

            # Check Schwartz (PVQ) correlations
            has_valid_pvq = False
            if 'correlations' in entry and entry['correlations']:
                for dimension, correlation in entry['correlations']:
                    logging.debug(f"PVQ - Dimension: {dimension}, Correlation: {correlation}")
                    if self._check_correlation(correlation) and not ((entry.get("portrait_id") == 3902 and entry.get("option_id") == 4 and dimension == "Achievement") or 
                                                                (entry.get("portrait_id") == 1901 and entry.get("option_id") == 1) and dimension == "Universalism"):
                        has_valid_pvq = True
                        if dimension not in dimension_scores['pvq']:
                            dimension_scores['pvq'][dimension] = []
                        score = numeric_response
                        if correlation < 0:
                            score = 7 - score
                            logging.debug(f"PVQ - Inverted score for {dimension}: {score}")
                        dimension_scores['pvq'][dimension].append(score)
                        logging.debug(f"PVQ - Added score {score} to dimension {dimension}")
            if has_valid_pvq:
                valid_responses['pvq'].append(numeric_response)
                logging.debug(f"Added response {numeric_response} to valid PVQ responses")

            # Check BFI correlations
            has_valid_bfi = False
            if 'bfi_correlations' in entry and entry['bfi_correlations']:
                for dimension, correlation in entry['bfi_correlations']:
                    logging.debug(f"BFI - Dimension: {dimension}, Correlation: {correlation}")
                    if self._check_correlation(correlation) and not (entry.get("portrait_id") == 2901 and entry.get("option_id") == 4 and dimension == "Extraversion"):
                        has_valid_bfi = True
                        if dimension not in dimension_scores['bfi']:
                            dimension_scores['bfi'][dimension] = []
                        score = numeric_response
                        if correlation < 0:
                            score = 7 - score
                            logging.debug(f"BFI - Inverted score for {dimension}: {score}")
                        dimension_scores['bfi'][dimension].append(score)
                        logging.debug(f"BFI - Added score {score} to dimension {dimension}")
            if has_valid_bfi:
                valid_responses['bfi'].append(numeric_response)
                logging.debug(f"Added response {numeric_response} to valid BFI responses")

            # Check Higher PVQ correlations
            has_valid_higher_pvq = False
            if 'higher_pvq_correlations' in entry and entry['higher_pvq_correlations']:
                for dimension, correlation in entry['higher_pvq_correlations']:
                    logging.debug(f"Higher PVQ - Dimension: {dimension}, Correlation: {correlation}")
                    if self._check_correlation(correlation):
                        has_valid_higher_pvq = True
                        if dimension not in dimension_scores['higher_pvq']:
                            dimension_scores['higher_pvq'][dimension] = []
                        score = numeric_response
                        if correlation < 0:
                            score = 7 - score
                            logging.debug(f"Higher PVQ - Inverted score for {dimension}: {score}")
                        dimension_scores['higher_pvq'][dimension].append(score)
                        logging.debug(f"Higher PVQ - Added score {score} to dimension {dimension}")
            if has_valid_higher_pvq:
                valid_responses['higher_pvq'].append(numeric_response)
                logging.debug(f"Added response {numeric_response} to valid Higher PVQ responses")

        # Calculate stats for each category and dimension
        logging.info("\nCalculating final statistics:")
        results = {}
        
        for category in ['pvq', 'bfi', 'higher_pvq']:
            results[category] = {}
            
            # Calculate category mean from valid responses
            category_mean = 0
            if valid_responses[category]:
                category_mean = sum(valid_responses[category]) / len(valid_responses[category])
                logging.info(f"\n{category.upper()} category mean (from {len(valid_responses[category])} valid responses): {category_mean:.2f}")
            
            # Process each dimension in the category
            for dimension, scores in dimension_scores[category].items():
                if not scores:
                    continue
                    
                mean = sum(scores) / len(scores)
                squared_diff_sum = sum((x - mean) ** 2 for x in scores)
                std_dev = math.sqrt(squared_diff_sum / len(scores))
                
                logging.info(f"\n{category.upper()} - {dimension}:")
                logging.info(f"Raw scores: {scores}")
                logging.info(f"Initial mean: {mean:.2f}")
                
                # If centering is enabled, subtract the category mean
                if self.center:
                    original_mean = mean
                    mean -= category_mean
                    logging.info(f"Centered mean: {mean:.2f} (original: {original_mean:.2f} - category_mean: {category_mean:.2f})")
                
                    results[category][dimension] = {
                        'centered_mean': mean,
                        'original_mean': original_mean,
                        'std_dev': std_dev,
                        'n': len(scores)
                    }
                else:
                    results[category][dimension] = {
                        'mean': mean,
                        'std_dev': std_dev,
                        'n': len(scores)
                    }
                logging.info(f"Final stats for {dimension}: {results[category][dimension]}")

        return results

    def save_results(self):
        logging.info("\nSaving results...")
        results = {
            'meta': {
                'model_name': self.model_name,
                'prompt': self.prompt + "_" + str(self.threshold) + "_" + self.cor_mode + ("_centered" if self.center else "")
            },
            'scores': self.calculate_dimension_scores()
        }
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Results saved to: {self.output_file}")
        return self.output_file

def process_directory(input_dir: str, output_dir: str, threshold: float, cor_mode: str = 'pos', center: bool = False) -> List[str]:
    logging.info(f"\nProcessing directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    saved_paths = []
    
    if not os.path.exists(input_dir):
        logging.error(f"Error: Input directory {input_dir} does not exist!")
        return saved_paths
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('results.json'):
                logging.info(f"\nProcessing file: {filename}")
                
                rel_path = os.path.relpath(root, input_dir)
                input_path = os.path.join(root, filename)
                suffix = "_centered" if center else ""
                output_filename = os.path.splitext(filename)[0] + "_" + str(threshold) + "_" + cor_mode + suffix + '_scores.json'
                output_path = os.path.join(output_dir, rel_path, output_filename)
                
                try:
                    scorer = DimensionScorer(input_path, output_path, threshold, cor_mode, center)
                    saved_path = scorer.save_results()
                    saved_paths.append(saved_path)
                    logging.info(f"Successfully processed: {input_path}")
                except Exception as e:
                    logging.error(f"Error processing {input_path}: {str(e)}")
    
    return saved_paths

def main():
    input_dir = "average_outputs"
    threshold = 0.3
    cor_mode = 'pos'
    center = True
    suffix = "_centered" if center else ""
    output_dir = f"score_results/final_results_{threshold}_{cor_mode}{suffix}"

    logging.info("\nStarting dimension scoring process...")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Threshold: {threshold}")
    logging.info(f"Correlation mode: {cor_mode}")
    logging.info(f"Centering: {center}")

    saved_paths = process_directory(input_dir, output_dir, threshold, cor_mode, center)
    
    logging.info(f"\nProcessing complete!")
    logging.info(f"Number of files processed: {len(saved_paths)}")
    logging.info("\nResults saved to:")
    for path in saved_paths:
        logging.info(f"- {path}")

if __name__ == "__main__":
    main()
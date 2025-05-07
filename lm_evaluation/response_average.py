import json
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

class ResponseAverager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        print(f"\nSearching for files in: {self.base_dir}")
        self.model_files = self._organize_files()
        
    def _get_base_model_name(self, filename: str) -> str:
        """Extract base model name from filename by splitting on underscores"""
        base_name = filename.rsplit('.', 1)[0]
        parts = base_name.split('_')
        base_parts = []
        for part in parts:
            if part.startswith('v') and part[1:].isdigit():
                break
            base_parts.append(part)
        return '_'.join(base_parts)

    def _organize_files(self) -> Dict[str, List[str]]:
        model_files = defaultdict(list)
        for root, _, files in os.walk(self.base_dir):
            for filename in files:
                if not filename.endswith('.json'):
                    continue
                base_model = self._get_base_model_name(filename)
                if not base_model:
                    continue
                full_path = os.path.join(root, filename)
                model_files[base_model].append(full_path)
        
        for model in model_files:
            model_files[model].sort()
            
        return dict(model_files)

    def _load_single_file(self, filepath: str) -> List[dict]:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error reading file {filepath}: {str(e)}")
            return []

    def _extract_numeric_response(self, entry: dict) -> Tuple[Optional[float], Optional[dict]]:
        """Extract numeric response and error if present"""
        if 'error' in entry:
            return None, {'error': entry['error']}
        try:
            response = entry.get('numeric_response')
            if response is not None:
                return float(response), None
        except (ValueError, TypeError):
            pass
        return None, None

    def _get_version_info(self, filepath: str) -> str:
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.startswith('v') and part[1:].isdigit():
                if i + 1 < len(parts) and parts[i + 1] == 'reversed':
                    return f"{part}_{parts[i + 1]}"
                return part
        return 'unknown'

    def calculate_model_averages(self, model_name: str) -> Tuple[List[dict], dict]:
        response_groups = defaultdict(lambda: {
            'responses': [], 
            'version_responses': {}, 
            'version_errors': {},
            'entry_template': None
        })
        
        total_errors = 0
        error_counts_by_version = defaultdict(int)
        
        # Process all version files for this model
        for filepath in self.model_files[model_name]:
            data = self._load_single_file(filepath)
            version_info = self._get_version_info(filepath)
            
            for entry in data:
                key = (entry['portrait_id'], entry.get('option_id', 1))
                numeric_response, error = self._extract_numeric_response(entry)
                
                if response_groups[key]['entry_template'] is None:
                    response_groups[key]['entry_template'] = entry
                
                if numeric_response is not None:
                    response_groups[key]['responses'].append(numeric_response)
                    response_groups[key]['version_responses'][version_info] = numeric_response
                elif error is not None:
                    response_groups[key]['version_errors'][version_info] = error
                    total_errors += 1
                    error_counts_by_version[version_info] += 1

        # Calculate averages and create output entries
        averaged_results = []
        for (portrait_id, option_id), group_data in response_groups.items():
            if not group_data['entry_template']:
                continue

            template = group_data['entry_template']
            responses = group_data['responses']
            version_responses = group_data['version_responses']
            version_errors = group_data['version_errors']
            
            averaged_entry = {
                'portrait_id': template['portrait_id'],
                'option_id': template.get('option_id', 1),
                'content': template['content'],
                'prompt': template['prompt'],
                'version_responses': version_responses,
            }
            
            # Add numeric_response only if we have valid responses
            if responses:
                averaged_entry['numeric_response'] = float(np.mean(responses))
            
            # Add version errors if any exist
            if version_errors:
                averaged_entry['version_errors'] = version_errors

            # Add correlation fields if they exist in the template
            if 'correlations' in template:
                averaged_entry['correlations'] = template['correlations']
            if 'bfi_correlations' in template:
                averaged_entry['bfi_correlations'] = template['bfi_correlations']
            if 'higher_pvq_correlations' in template:
                averaged_entry['higher_pvq_correlations'] = template['higher_pvq_correlations']
            
            averaged_results.append(averaged_entry)
            
        averaged_results.sort(key=lambda x: (x['portrait_id'], x['option_id']))
        
        # Create metadata about errors
        error_metadata = {
            'total_errors': total_errors,
            'errors_by_version': dict(error_counts_by_version)
        }
        
        return averaged_results, error_metadata

    def save_all_model_averages(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.model_files:
            print("\nNo model files found to process!")
            return
            
        print("\nProcessing models:")
        for model_name in self.model_files:
            try:
                print(f"\nModel: {model_name}")
                print(f"Found {len(self.model_files[model_name])} version files")
                
                averaged_results, error_metadata = self.calculate_model_averages(model_name)
                
                # Save results
                results_filename = f"{model_name}_averaged_results.json"
                results_path = os.path.join(output_dir, results_filename)
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(averaged_results, f, indent=2, ensure_ascii=False)

                # Save metadata
                metadata_filename = f"{model_name}_metadata.json"
                metadata_path = os.path.join(output_dir, metadata_filename)
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(error_metadata, f, indent=2, ensure_ascii=False)
                
                print(f"Items processed: {len(averaged_results)}")
                print(f"Total errors: {error_metadata['total_errors']}")
                print(f"Results saved to: {results_path}")
                print(f"Metadata saved to: {metadata_path}")
                
            except Exception as e:
                print(f"Error processing model {model_name}: {str(e)}")

def main():
    base_dir = "outputs/final"
    output_dir = "average_outputs"
    
    print("\nStarting processing...")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    averager = ResponseAverager(base_dir)
    averager.save_all_model_averages(output_dir)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
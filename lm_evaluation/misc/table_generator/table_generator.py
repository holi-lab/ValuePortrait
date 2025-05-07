import json
import os
from pathlib import Path

# Define dimension orders
PVQ_ORDER = [
    'Universalism', 'Benevolence', 'Conformity', 'Tradition', 'Security',
    'Power', 'Achievement', 'Hedonism', 'Stimulation', 'Self-Direction'
]

BFI_ORDER = [
    'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'
]

# Define model family order
MODEL_FAMILY_ORDER = {
    'gpt': 1,
    'claude': 2,
    'qwen': 3,
    'mistral': 4,
    'llama': 5,
    'deepseek': 6,
    'grok': 7
}

def get_model_family(model_name):
    model_name = model_name.lower()
    # Special handling for o1-mini and o3-mini
    if model_name.startswith('o1-mini') or model_name.startswith('o3-mini'):
        return 'gpt'
    for family in MODEL_FAMILY_ORDER.keys():
        if family in model_name:
            return family
    return 'other'

def model_sort_key(model_data):
    model_name = model_data['meta']['model_name'].lower()
    family = get_model_family(model_name)
    family_order = MODEL_FAMILY_ORDER.get(family, 999)  # Unknown families go to the end
    return (family_order, model_name)

def create_pretty_table_from_jsons(directory_path, mode='pvq'):
    # Get all JSON files in the directory
    json_files = list(Path(directory_path).glob('*.json'))
    
    # Store all data
    all_data = []
    
    # Read each JSON file
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_data.append(data)
    
    if not all_data:
        return "No JSON files found in the directory."
    
    # Sort data by model family and then by model name
    all_data.sort(key=model_sort_key)
    
    # Set dimensions and value type based on mode
    if mode.lower() == 'pvq':
        dimensions = PVQ_ORDER
        value_type = 'centered_mean'
    else:  # bfi mode
        dimensions = BFI_ORDER
        value_type = 'original_mean'
    
    # Calculate column widths
    col_widths = {'model': max(len('Model'), max(len(d['meta']['model_name']) for d in all_data))}
    for dim in dimensions:
        col_widths[dim] = max(len(dim), 8)  # 8 for "-0.00" format (changed from 8 to 7)
    
    # Create header
    header = '| Model ' + ' ' * (col_widths['model'] - 5)
    for dim in dimensions:
        header += f'| {dim:<{col_widths[dim]}} '
    header += '|'
    
    # Create separator
    separator = '|-' + '-' * (col_widths['model'])
    for dim in dimensions:
        separator += '|-' + '-' * col_widths[dim]
    separator += '|'
    
    # Create rows
    rows = []
    current_family = None
    for data in all_data:
        model_name = data['meta']['model_name']
        family = get_model_family(model_name)
        
        # Add empty row between different families
        if current_family is not None and current_family != family:
            rows.append('|' + '-' * (len(header) - 2) + '|')
        current_family = family
        
        row = f"| {model_name:<{col_widths['model']}} "
        for dim in dimensions:
            # Handle Self-Direction special case for PVQ mode
            lookup_dim = 'Self_Direction' if dim == 'Self-Direction' and mode.lower() == 'pvq' else dim
            try:
                value = data['scores'][mode.lower()][lookup_dim][value_type]
                # Apply scaling for BFI mode
                if mode.lower() == 'bfi':
                    value = value * (5/6)
                # Round to 2 decimal places
                value = round(value, 2)
                row += f"| {value:>{col_widths[dim]}.2f} "
            except KeyError:
                row += f"| {'N/A':<{col_widths[dim]}} "
        row += '|'
        rows.append(row)
    
    # Combine all parts
    table = f"{header}\n{separator}\n" + "\n".join(rows)
    
    return table

def save_table_to_file(table_str, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(table_str)

# Example usage
if __name__ == "__main__":
    # Directory containing JSON files
    directory_path = "score_results/final_results_0.3_pos_centered"
    
    # Create PVQ table
    pvq_table = create_pretty_table_from_jsons(directory_path, mode='pvq')
    save_table_to_file(pvq_table, 'error_fixed_pvq_values.txt')
    print("PVQ Table:")
    print(pvq_table)
    print("\n")
    
    # Create BFI table
    bfi_table = create_pretty_table_from_jsons(directory_path, mode='bfi')
    save_table_to_file(bfi_table, 'error_fixed_bfi_values.txt')
    print("BFI Table:")
    print(bfi_table)
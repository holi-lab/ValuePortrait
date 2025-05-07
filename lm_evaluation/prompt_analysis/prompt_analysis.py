import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import glob
from pathlib import Path

def calculate_version_differences(values: List[float]) -> float:
    """
    Calculate the average pairwise difference between values.
    """
    if len(values) < 2:
        return 0

    differences = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            diff = abs(values[i] - values[j])
            differences.append(diff)

    return np.mean(differences) if differences else 0

def calculate_sensitivities(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate sensitivities and difference metrics.
    Now, it also calculates global_avg_version_difference for all responses 
    (v1, v2, v3, v1_reversed, v2_reversed, v3_reversed).
    """
    version_results = {
        'v1': {'valid_items': [], 'agreements': [], 'order_differences': []},
        'v2': {'valid_items': [], 'agreements': [], 'order_differences': []},
        'v3': {'valid_items': [], 'agreements': [], 'order_differences': []}
    }

    prompt_agreements = []
    version_differences = []           # for v1, v2, v3
    global_version_differences = []    # for all responses
    skipped_items = []

    for idx, item in enumerate(data):
        try:
            versions_present = []
            version_values = []
            all_values = []

            for version in ['v1', 'v2', 'v3']:
                if check_valid_item(item, version):
                    versions_present.append(version)
                    version_values.append(item['version_responses'][version])
                    all_values.append(item['version_responses'][version])
                    all_values.append(item['version_responses'][f'{version}_reversed'])

            if len(versions_present) >= 2:
                si = 1 if len(set(version_values)) == 1 else 0
                prompt_agreements.append(si)
                version_diff = calculate_version_differences(version_values)
                version_differences.append(version_diff)

            if len(all_values) >= 2:
                global_diff = calculate_version_differences(all_values)
                global_version_differences.append(global_diff)

            for version in ['v1', 'v2', 'v3']:
                if check_valid_item(item, version):
                    version_results[version]['valid_items'].append(item)

                    orig_value = item['version_responses'][version]
                    rev_value = item['version_responses'][f'{version}_reversed']

                    agreement = 1 if orig_value == rev_value else 0
                    version_results[version]['agreements'].append(agreement)

                    order_diff = abs(orig_value - rev_value)
                    version_results[version]['order_differences'].append(order_diff)

        except Exception as e:
            skipped_items.append({
                'index': idx,
                'portrait_id': item.get('portrait_id', 'unknown'),
                'error': str(e)
            })
            continue

    order_sensitivities = {}
    for version in ['v1', 'v2', 'v3']:
        n_valid = len(version_results[version]['agreements'])
        if n_valid > 0:
            sensitivity = 1 - (1 / n_valid) * sum(version_results[version]['agreements'])
            avg_difference = np.mean(version_results[version]['order_differences'])
            order_sensitivities[version] = {
                'sensitivity': sensitivity,
                'avg_difference': avg_difference,
                'n_valid': n_valid
            }
        else:
            order_sensitivities[version] = {
                'sensitivity': None,
                'avg_difference': None,
                'n_valid': 0
            }

    valid_sensitivities = [
        order_sensitivities[v]['sensitivity']
        for v in ['v1', 'v2', 'v3']
        if order_sensitivities[v]['sensitivity'] is not None
    ]

    valid_differences = [
        order_sensitivities[v]['avg_difference']
        for v in ['v1', 'v2', 'v3']
        if order_sensitivities[v]['avg_difference'] is not None
    ]

    avg_order_sensitivity = np.mean(valid_sensitivities) if valid_sensitivities else None
    avg_order_difference = np.mean(valid_differences) if valid_differences else None

    n_prompt_valid = len(prompt_agreements)
    if n_prompt_valid > 0:
        prompt_sensitivity = 1 - (1 / n_prompt_valid) * sum(prompt_agreements)
        avg_version_difference = np.mean(version_differences)
    else:
        prompt_sensitivity = None
        avg_version_difference = None

    avg_global_version_difference = np.mean(global_version_differences) if global_version_differences else None

    results = {
        'prompt_metrics': {
            'sensitivity': prompt_sensitivity,
            'avg_version_difference': avg_version_difference,
            'global_avg_version_difference': avg_global_version_difference
        },
        'order_metrics': {
            'sensitivities': {
                'v1': order_sensitivities['v1']['sensitivity'],
                'v2': order_sensitivities['v2']['sensitivity'],
                'v3': order_sensitivities['v3']['sensitivity'],
                'average': avg_order_sensitivity
            },
            'differences': {
                'v1': order_sensitivities['v1']['avg_difference'],
                'v2': order_sensitivities['v2']['avg_difference'],
                'v3': order_sensitivities['v3']['avg_difference'],
                'average': avg_order_difference
            }
        },
        'details': {
            'total_items': len(data),
            'valid_items_by_version': {
                'v1': order_sensitivities['v1']['n_valid'],
                'v2': order_sensitivities['v2']['n_valid'],
                'v3': order_sensitivities['v3']['n_valid']
            },
            'prompt_valid_items': n_prompt_valid,
            'skipped_items': skipped_items
        }
    }

    return results

def calculate_consistency(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate the consistency metric across items.
    For each prompt version (v1, v1_reversed, v2, v2_reversed, v3, v3_reversed),
    compute the standard deviation of responses over all items.
    Then, compute the overall consistency as the average of these six standard deviations.
    """
    prompt_versions = ['v1', 'v1_reversed', 'v2', 'v2_reversed', 'v3', 'v3_reversed']
    responses = {version: [] for version in prompt_versions}

    for item in data:
        version_responses = item.get('version_responses', {})
        for version in prompt_versions:
            if version in version_responses:
                responses[version].append(version_responses[version])

    per_prompt_std = {}
    for version in prompt_versions:
        if responses[version]:
            std_val = np.std(responses[version])
        else:
            std_val = None
        per_prompt_std[version] = std_val

    valid_std = [val for val in per_prompt_std.values() if val is not None]
    overall_consistency = np.mean(valid_std) if valid_std else None

    return {
        'per_prompt_std': per_prompt_std,
        'overall_consistency': overall_consistency
    }

def print_directory_report(results: Dict[str, Any]) -> None:
    print("\n=== Directory Processing Report ===")
    print(f"Base Directory: {results['base_directory']}")
    print(f"Timestamp: {results['timestamp']}")

    summary = results['summary']
    print(f"\nSummary:")
    print(f"Total files processed: {summary['total_files']}")
    print(f"Successful files: {summary['successful_files']}")
    print(f"Failed files: {summary['failed_files']}")

    if summary['successful_files'] > 0:
        print(f"\nPrompt Metrics:")
        print(f"Average Sensitivity: {summary['average_prompt_metrics']['sensitivity']:.3f}")
        print(f"Average Version Difference (v1,v2,v3): {summary['average_prompt_metrics']['avg_version_difference']:.3f}")
        print(f"Global Average Version Difference (all responses): {summary['average_prompt_metrics']['global_avg_version_difference']:.3f}")

        print(f"\nOrder Metrics:")
        print("Sensitivities by Version:")
        for version in ['v1', 'v2', 'v3']:
            print(f"{version}: {summary['average_order_metrics']['sensitivities'][version]:.3f}")
        print(f"Overall Average Sensitivity: {summary['average_order_metrics']['sensitivities']['average']:.3f}")

        print("\nAverage Differences by Version:")
        for version in ['v1', 'v2', 'v3']:
            print(f"{version}: {summary['average_order_metrics']['differences'][version]:.3f}")
        print(f"Overall Average Difference: {summary['average_order_metrics']['differences']['average']:.3f}")

        print("\nConsistency Metrics:")
        print("Average Standard Deviation per Prompt Version:")
        for version, std_val in summary['average_consistency_metrics']['per_prompt_std'].items():
            print(f"{version}: {std_val:.3f}" if std_val is not None else f"{version}: N/A")
        print(f"Overall Consistency (average std): {summary['average_consistency_metrics']['overall_consistency']:.3f}")

    if results['files_failed']:
        print(f"\nFailed Files:")
        for failure in results['files_failed']:
            print(f"- {failure['file_path']}: {failure['error']}")

def check_valid_item(item: Dict[str, Any], version: str) -> bool:
    try:
        version_responses = item.get('version_responses', {})
        required_keys = {f'{version}', f'{version}_reversed'}
        return all(key in version_responses for key in required_keys)
    except:
        return False

def process_directory(dir_path: str, output_dir: str = None) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = 'sensitivity_results'

    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'base_directory': dir_path,
        'files_processed': [],
        'files_failed': [],
        'summary': {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'average_prompt_metrics': {
                'sensitivity': 0,
                'avg_version_difference': 0,
                'global_avg_version_difference': 0
            },
            'average_order_metrics': {
                'sensitivities': {
                    'v1': 0,
                    'v2': 0,
                    'v3': 0,
                    'average': 0
                },
                'differences': {
                    'v1': 0,
                    'v2': 0,
                    'v3': 0,
                    'average': 0
                }
            },
            'average_consistency_metrics': {
                'per_prompt_std': {
                    'v1': 0,
                    'v1_reversed': 0,
                    'v2': 0,
                    'v2_reversed': 0,
                    'v3': 0,
                    'v3_reversed': 0
                },
                'overall_consistency': 0
            }
        }
    }

    json_files = glob.glob(os.path.join(dir_path, '**', '*results.json'), recursive=True)
    all_results['summary']['total_files'] = len(json_files)

    prompt_sensitivities = []
    original_version_differences = []
    global_version_differences = []
    order_sensitivities_by_version = {'v1': [], 'v2': [], 'v3': []}
    order_differences_by_version = {'v1': [], 'v2': [], 'v3': []}
    consistency_per_prompt = {
        'v1': [], 'v1_reversed': [], 'v2': [], 'v2_reversed': [], 'v3': [], 'v3_reversed': []
    }
    overall_consistency_list = []

    for file_path in json_files:
        relative_path = os.path.relpath(file_path, dir_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = calculate_sensitivities(data)
            # Calculate consistency metrics for the same data
            consistency_results = calculate_consistency(data)
            results['consistency_metrics'] = consistency_results

            output_filename = f"{Path(relative_path).stem}_results.json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            if results['prompt_metrics']['sensitivity'] is not None:
                prompt_sensitivities.append(results['prompt_metrics']['sensitivity'])
            if results['prompt_metrics']['avg_version_difference'] is not None:
                original_version_differences.append(results['prompt_metrics']['avg_version_difference'])
            if results['prompt_metrics']['global_avg_version_difference'] is not None:
                global_version_differences.append(results['prompt_metrics']['global_avg_version_difference'])

            for version in ['v1', 'v2', 'v3']:
                sens = results['order_metrics']['sensitivities'][version]
                diff = results['order_metrics']['differences'][version]
                if sens is not None:
                    order_sensitivities_by_version[version].append(sens)
                if diff is not None:
                    order_differences_by_version[version].append(diff)

            # Collect consistency metrics
            for version, std_val in results['consistency_metrics']['per_prompt_std'].items():
                if std_val is not None:
                    consistency_per_prompt[version].append(std_val)
            if results['consistency_metrics']['overall_consistency'] is not None:
                overall_consistency_list.append(results['consistency_metrics']['overall_consistency'])

            all_results['files_processed'].append({
                'file_path': relative_path,
                'results': results,
                'output_path': output_path
            })

            all_results['summary']['successful_files'] += 1

        except Exception as e:
            all_results['files_failed'].append({
                'file_path': relative_path,
                'error': str(e)
            })
            all_results['summary']['failed_files'] += 1

    avg_prompt_sensitivity = np.mean(prompt_sensitivities) if prompt_sensitivities else 0
    avg_original_version_difference = np.mean(original_version_differences) if original_version_differences else 0
    avg_global_version_difference = np.mean(global_version_differences) if global_version_differences else 0
    all_results['summary']['average_prompt_metrics'] = {
        'sensitivity': avg_prompt_sensitivity,
        'avg_version_difference': avg_original_version_difference,
        'global_avg_version_difference': avg_global_version_difference
    }

    avg_order_sensitivities = {}
    avg_order_differences = {}
    for version in ['v1', 'v2', 'v3']:
        avg_order_sensitivities[version] = np.mean(order_sensitivities_by_version[version]) if order_sensitivities_by_version[version] else 0
        avg_order_differences[version] = np.mean(order_differences_by_version[version]) if order_differences_by_version[version] else 0
    valid_sens_avgs = [avg_order_sensitivities[v] for v in ['v1', 'v2', 'v3'] if avg_order_sensitivities[v] != 0]
    overall_order_sensitivity = np.mean(valid_sens_avgs) if valid_sens_avgs else 0
    valid_diff_avgs = [avg_order_differences[v] for v in ['v1', 'v2', 'v3'] if avg_order_differences[v] != 0]
    overall_order_difference = np.mean(valid_diff_avgs) if valid_diff_avgs else 0

    all_results['summary']['average_order_metrics'] = {
        'sensitivities': {
            'v1': avg_order_sensitivities['v1'],
            'v2': avg_order_sensitivities['v2'],
            'v3': avg_order_sensitivities['v3'],
            'average': overall_order_sensitivity
        },
        'differences': {
            'v1': avg_order_differences['v1'],
            'v2': avg_order_differences['v2'],
            'v3': avg_order_differences['v3'],
            'average': overall_order_difference
        }
    }

    avg_consistency = {}
    for version in ['v1', 'v1_reversed', 'v2', 'v2_reversed', 'v3', 'v3_reversed']:
        avg_consistency[version] = np.mean(consistency_per_prompt[version]) if consistency_per_prompt[version] else 0
    overall_consistency = np.mean(overall_consistency_list) if overall_consistency_list else 0

    all_results['summary']['average_consistency_metrics'] = {
        'per_prompt_std': avg_consistency,
        'overall_consistency': overall_consistency
    }

    summary_path = os.path.join(output_dir, 'summary_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    return all_results

def main():
    try:
        dir_path = "average_outputs"
        results = process_directory(dir_path)
        print_directory_report(results)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
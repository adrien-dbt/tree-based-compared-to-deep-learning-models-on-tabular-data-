"""
Simple script to run benchmarks on multiple datasets.
This is a simplified version for quick experimentation.
"""

from benchmark import Benchmark
import json

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

# Define datasets to benchmark.
# n_samples_approx is used only for ordering — smallest datasets run first.
_DATASETS_UNORDERED = [
    # --- Classification ---
    {'name': 'credit-g',                               'description': 'German Credit (Classification)',           'n_samples_approx': 1000},
    {'name': 'diabetes',                               'description': 'Pima Indians Diabetes (Classification)',   'n_samples_approx': 768},
    {'dataset_id': 40536, 'name': 'speed_dating',      'description': 'SpeedDating (Classification)',             'n_samples_approx': 8380},
    {'dataset_id': 61,   'name': 'iris',               'description': 'Iris (Classification)',                    'n_samples_approx': 150},
    {'dataset_id': 40,   'name': 'sonar',              'description': 'Sonar (Classification)',                   'n_samples_approx': 208},
    {'dataset_id': 1464, 'name': 'blood-transfusion',  'description': 'Blood Transfusion (Classification)',       'n_samples_approx': 748},
    # --- Regression ---
    {'name': 'boston',                                 'description': 'Boston Housing (Regression)',              'n_samples_approx': 506},
    {'dataset_id': 41,   'name': 'servo',              'description': 'Servo (Regression)',                       'n_samples_approx': 167},
    {'dataset_id': 560,  'name': 'bodyfat',            'description': 'Body Fat (Regression)',                    'n_samples_approx': 252},
]

DATASETS = sorted(_DATASETS_UNORDERED, key=lambda d: d['n_samples_approx'])


def run_single_dataset(dataset_config, run_tree=True, run_deep=True):
    """
    Run benchmark on a single dataset.
    
    Args:
        dataset_config: Dictionary with dataset configuration
        run_tree: Whether to run tree-based models
        run_deep: Whether to run deep learning models
    
    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_config.get('description', 'Unknown')}")
    print("=" * 70)
    
    # Create benchmark
    if 'dataset_id' in dataset_config:
        benchmark = Benchmark(dataset_id=dataset_config['dataset_id'])
    elif 'name' in dataset_config:
        benchmark = Benchmark(dataset_name=dataset_config['name'])
    else:
        benchmark = Benchmark(
            suite_id=dataset_config.get('suite_id', 337),
            task_id=dataset_config.get('task_id')
        )
    
    # Run all models
    results_df = benchmark.run_all(run_tree=run_tree, run_deep=run_deep)
    
    # Print summary
    benchmark.print_summary()
    
    # Return results
    return {
        'dataset': dataset_config.get('name') or f"dataset_{dataset_config.get('dataset_id')}" or f"suite_{dataset_config.get('suite_id')}_task_{benchmark.task_id}",
        'description': dataset_config.get('description', ''),
        'task_type': benchmark.task_type,
        'n_samples_train': len(benchmark.X_train),
        'n_samples_test': len(benchmark.X_test),
        'n_features': len(benchmark.feature_names),
        'results': results_df.to_dict('records')
    }


def main():
    """
    Run benchmarks on multiple datasets and compare results.
    """
    print("=" * 70)
    print("MULTI-DATASET TABULAR DATA BENCHMARK")
    print("Based on: 'Why do tree-based models still outperform deep learning")
    print("           on typical tabular data?' (NeurIPS 2022)")
    print("=" * 70)
    
    output_file = 'multi_dataset_results.json'
    all_results = []

    # Run benchmark on each dataset
    for i, dataset_config in enumerate(DATASETS, 1):
        print(f"\n\n>>> Running dataset {i}/{len(DATASETS)}")

        try:
            result = run_single_dataset(dataset_config, run_tree=True, run_deep=True)
            all_results.append(result)
        except Exception as e:
            print(f"Error running dataset {dataset_config}: {e}")
            continue

        # Write results to file after each dataset completes
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n>>> Results saved to {output_file} ({len(all_results)}/{len(DATASETS)} datasets done)")

    # Print overall summary
    print("\n\n" + "=" * 70)
    print("OVERALL SUMMARY ACROSS ALL DATASETS")
    print("=" * 70)

    for result in all_results:
        print(f"\n--- {result['dataset']} ({result['description']}) ---")
        print(f"Samples: {result['n_samples_train']} train, {result['n_samples_test']} test")
        print(f"Features: {result['n_features']}")

        # Find best models
        df_results = result['results']

        # Get metric column
        if result['task_type'] == 'classification':
            metric_key = 'roc_auc' if any('roc_auc' in r for r in df_results) else 'accuracy'
        else:
            metric_key = 'r2' if any('r2' in r for r in df_results) else 'rmse'

        # Sort by metric
        sorted_results = sorted(df_results, key=lambda x: x.get(metric_key, 0),
                               reverse=(metric_key != 'rmse'))

        print(f"\nTop 3 models by {metric_key}:")
        for j, r in enumerate(sorted_results[:3], 1):
            print(f"  {j}. {r['model']} ({r['type']}): {metric_key}={r.get(metric_key, 'N/A'):.4f}, "
                  f"train_time={r['train_time']:.2f}s")

    print("\n\n" + "=" * 70)
    print(f"All results saved to {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

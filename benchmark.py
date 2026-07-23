"""
Main benchmark script for comparing tree-based models and deep learning on tabular data.
Based on the paper: "Why do tree-based models still outperform deep learning on typical tabular data?"
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any
import json

from data_loader import load_openml_dataset, get_dataset_by_name, get_dataset_by_id
from models.tree_models import RandomForest, XGBoost, LightGBM
from models.deep_models import MLP, TabNetModel, FTTransformer, SAINT


class Benchmark:
    """
    Benchmark class for comparing different models on tabular datasets.
    """
    
    def __init__(self, dataset_name: str = None, dataset_id: int = None, suite_id: int = 337, task_id: int = None):
        """
        Initialize benchmark.
        
        Args:
            dataset_name: Name of dataset to load from OpenML
            dataset_id: OpenML dataset ID (exact, preferred over name search)
            suite_id: OpenML benchmark suite ID
            task_id: Specific task ID from suite
        """
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.suite_id = suite_id
        self.task_id = task_id
        self.results = []

        # Load dataset
        print("=" * 60)
        print("LOADING DATASET")
        print("=" * 60)

        if dataset_id:
            self.X_train, self.X_test, self.y_train, self.y_test, self.cat_indicator, self.feature_names = \
                get_dataset_by_id(dataset_id)
        elif dataset_name:
            self.X_train, self.X_test, self.y_train, self.y_test, self.cat_indicator, self.feature_names = \
                get_dataset_by_name(dataset_name)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test, self.cat_indicator, self.feature_names = \
                load_openml_dataset(suite_id=suite_id, task_id=task_id)
        
        # Determine task type
        self.task_type = 'classification' if len(np.unique(self.y_train)) < 20 else 'regression'
        print(f"\nTask type: {self.task_type}")
        print(f"Number of classes/unique values: {len(np.unique(self.y_train))}")
    
    def run_tree_models(self) -> List[Dict[str, Any]]:
        """Run tree-based models."""
        print("\n" + "=" * 60)
        print("RUNNING TREE-BASED MODELS")
        print("=" * 60)
        
        models = {
            'RandomForest': RandomForest(self.task_type),
            'XGBoost': XGBoost(self.task_type),
            'LightGBM': LightGBM(self.task_type)
        }
        
        results = []
        
        for name, model in models.items():
            print(f"\n--- {name} ---")
            
            # Train
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            start_time = time.time()
            metrics = model.evaluate(self.X_test, self.y_test)
            inference_time = time.time() - start_time
            
            result = {
                'model': name,
                'type': 'tree-based',
                'train_time': train_time,
                'inference_time': inference_time,
                **metrics
            }
            results.append(result)
            
            print(f"Train time: {train_time:.2f}s, Inference time: {inference_time:.4f}s")
            print(f"Metrics: {metrics}")
        
        return results
    
    def run_deep_models(self) -> List[Dict[str, Any]]:
        """Run deep learning models."""
        print("\n" + "=" * 60)
        print("RUNNING DEEP LEARNING MODELS")
        print("=" * 60)
        
        models = {
            'MLP': MLP(
                task_type=self.task_type,
                hidden_dims=[128, 64],
                epochs=100,
                early_stopping_patience=10
            )
        }
        
        # Try to add TabNet if available
        try:
            models['TabNet'] = TabNetModel(
                task_type=self.task_type,
                max_epochs=100,
                patience=10
            )
        except ImportError:
            print("TabNet not available, skipping...")

        models['FT-Transformer'] = FTTransformer(
            task_type=self.task_type,
            d_token=64, n_heads=8, n_layers=3,
            epochs=100, early_stopping_patience=10
        )
        models['SAINT'] = SAINT(
            task_type=self.task_type,
            d_token=32, n_heads=4, n_layers=2,
            epochs=100, early_stopping_patience=10
        )
        
        results = []
        
        for name, model in models.items():
            print(f"\n--- {name} ---")
            
            # Train
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            start_time = time.time()
            metrics = model.evaluate(self.X_test, self.y_test)
            inference_time = time.time() - start_time
            
            result = {
                'model': name,
                'type': 'deep-learning',
                'train_time': train_time,
                'inference_time': inference_time,
                **metrics
            }
            results.append(result)
            
            print(f"Train time: {train_time:.2f}s, Inference time: {inference_time:.4f}s")
            print(f"Metrics: {metrics}")
        
        return results
    
    def run_all(self, run_tree: bool = True, run_deep: bool = True) -> pd.DataFrame:
        """Run all models and return results."""
        all_results = []
        
        if run_tree:
            all_results.extend(self.run_tree_models())
        
        if run_deep:
            all_results.extend(self.run_deep_models())
        
        # Create results DataFrame
        self.results_df = pd.DataFrame(all_results)
        
        return self.results_df
    
    def print_summary(self):
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        if len(self.results_df) == 0:
            print("No results to display.")
            return
        
        # Sort by primary metric
        if self.task_type == 'classification':
            metric_col = 'roc_auc' if 'roc_auc' in self.results_df.columns else 'accuracy'
        else:
            metric_col = 'r2' if 'r2' in self.results_df.columns else 'rmse'
        
        print(f"\nSorted by {metric_col}:")
        print(self.results_df.sort_values(metric_col, ascending=(metric_col == 'rmse')))
        
        # Compare tree vs deep learning
        print("\n" + "-" * 60)
        print("TREE-BASED vs DEEP LEARNING COMPARISON")
        print("-" * 60)
        
        tree_results = self.results_df[self.results_df['type'] == 'tree-based']
        deep_results = self.results_df[self.results_df['type'] == 'deep-learning']
        
        if len(tree_results) > 0 and len(deep_results) > 0:
            tree_best = tree_results.loc[tree_results[metric_col].idxmax() if metric_col != 'rmse' else tree_results[metric_col].idxmin()]
            deep_best = deep_results.loc[deep_results[metric_col].idxmax() if metric_col != 'rmse' else deep_results[metric_col].idxmin()]
            
            print(f"\nBest Tree-based Model: {tree_best['model']}")
            print(f"  {metric_col}: {tree_best[metric_col]:.4f}")
            print(f"  Train time: {tree_best['train_time']:.2f}s")
            
            print(f"\nBest Deep Learning Model: {deep_best['model']}")
            print(f"  {metric_col}: {deep_best[metric_col]:.4f}")
            print(f"  Train time: {deep_best['train_time']:.2f}s")
            
            # Performance difference
            if metric_col != 'rmse':
                diff = tree_best[metric_col] - deep_best[metric_col]
                print(f"\nPerformance gap (Tree - Deep): {diff:.4f}")
                if diff > 0:
                    print("Tree-based model performs better")
                else:
                    print("Deep learning model performs better")
            else:
                diff = deep_best[metric_col] - tree_best[metric_col]
                print(f"\nPerformance gap (Deep - Tree): {diff:.4f}")
                if diff > 0:
                    print("Tree-based model performs better (lower RMSE)")
                else:
                    print("Deep learning model performs better (lower RMSE)")
    
    def save_results(self, filename: str = 'benchmark_results.json'):
        """Save results to file."""
        results_dict = {
            'dataset': self.dataset_name or f"suite_{self.suite_id}_task_{self.task_id}",
            'task_type': self.task_type,
            'n_samples_train': len(self.X_train),
            'n_samples_test': len(self.X_test),
            'n_features': len(self.feature_names),
            'results': self.results_df.to_dict('records')
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {filename}")


def main():
    """Main function to run benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tabular Data Benchmark')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name from OpenML')
    parser.add_argument('--suite', type=int, default=337, help='OpenML suite ID (default: 337)')
    parser.add_argument('--task', type=int, default=None, help='Specific task ID')
    parser.add_argument('--no-tree', action='store_true', help='Skip tree-based models')
    parser.add_argument('--no-deep', action='store_true', help='Skip deep learning models')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = Benchmark(
        dataset_name=args.dataset,
        suite_id=args.suite,
        task_id=args.task
    )
    
    results = benchmark.run_all(
        run_tree=not args.no_tree,
        run_deep=not args.no_deep
    )
    
    benchmark.print_summary()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    # Example usage without command line arguments
    print("Running example benchmark...")
    
    # Use a simple dataset for demonstration
    benchmark = Benchmark(suite_id=337)  # Classification with numerical features
    results = benchmark.run_all()
    benchmark.print_summary()

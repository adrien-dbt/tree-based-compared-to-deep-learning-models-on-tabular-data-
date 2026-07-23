"""
Data loading utilities for tabular data benchmark.
Downloads datasets from OpenML benchmark suites used in the paper.
"""

import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Optional, List


def load_openml_dataset(
    suite_id: int = 337,  # Classification on numerical features
    task_id: Optional[int] = None,
    dataset_name: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[bool], List[str]]:
    """
    Load a dataset from OpenML benchmark suite.
    
    Args:
        suite_id: OpenML suite ID (334: clf num+cat, 335: reg num+cat, 337: clf num only)
        task_id: Specific task ID (if None, uses first task in suite)
        dataset_name: Filter by dataset name
        test_size: Fraction of data for test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, categorical_indicator, feature_names
    """
    # Get benchmark suite
    benchmark_suite = openml.study.get_suite(suite_id)
    
    # Select task
    if task_id is None:
        task_id = benchmark_suite.tasks[0]
    
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    
    print(f"Loading dataset: {dataset.name}")
    print(f"Dataset ID: {dataset.dataset_id}")
    
    # Get data
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )
    
    # Filter by name if specified
    if dataset_name and dataset.name != dataset_name:
        print(f"Warning: Dataset name mismatch. Expected {dataset_name}, got {dataset.name}")
    
    # Encode target if categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=y.name)
        print(f"Target encoded. Classes: {le.classes_}")
    
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    for col in X.select_dtypes(include=['object', 'category']).columns:
        # Convert to string first to avoid category issues
        X[col] = X[col].astype(str).fillna('missing')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) < 100 else None
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features: {len(attribute_names)}, Categorical: {sum(categorical_indicator)}")
    
    return X_train, X_test, y_train, y_test, categorical_indicator, attribute_names


def get_dataset_by_name(name: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load a specific dataset by name from OpenML.
    Popular datasets from the paper: 'credit-g', 'diabetes', 'vehicle', 'kc1'
    """
    # Search for dataset
    datasets = openml.datasets.list_datasets(output_format='dataframe')
    matching = datasets[datasets['name'].str.contains(name, case=False, na=False)]
    
    if len(matching) == 0:
        raise ValueError(f"Dataset '{name}' not found on OpenML")
    
    # Use first match
    dataset_id = int(matching.iloc[0]['did'])
    dataset = openml.datasets.get_dataset(dataset_id)
    
    print(f"Loading dataset: {dataset.name} (ID: {dataset_id})")
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )
    
    # Encode target
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name='target')
    
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    for col in X.select_dtypes(include=['object', 'category']).columns:
        # Convert to string first to avoid category issues
        X[col] = X[col].astype(str).fillna('missing')
    
    # Split
    stratify = y if len(np.unique(y)) < 100 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    return X_train, X_test, y_train, y_test, categorical_indicator, attribute_names


def get_dataset_by_id(dataset_id: int, test_size: float = 0.2, random_state: int = 42):
    """Load a specific dataset by its OpenML dataset ID."""
    dataset = openml.datasets.get_dataset(dataset_id)

    print(f"Loading dataset: {dataset.name} (ID: {dataset_id})")

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )

    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name='target')
        print(f"Target encoded. Classes: {le.classes_}")

    X = X.fillna(X.median(numeric_only=True))
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype(str).fillna('missing')

    stratify = y if len(np.unique(y)) < 100 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features: {len(attribute_names)}, Categorical: {sum(categorical_indicator)}")

    return X_train, X_test, y_train, y_test, categorical_indicator, attribute_names


def list_available_datasets(suite_id: int = 337) -> pd.DataFrame:
    """List all datasets available in a benchmark suite."""
    benchmark_suite = openml.study.get_suite(suite_id)
    
    datasets_info = []
    for task_id in benchmark_suite.tasks[:10]:  # Limit to first 10
        try:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            datasets_info.append({
                'task_id': task_id,
                'dataset_id': dataset.dataset_id,
                'name': dataset.name,
                'n_features': len(dataset.features) - 1  # Exclude target
            })
        except Exception as e:
            print(f"Error loading task {task_id}: {e}")
    
    return pd.DataFrame(datasets_info)


if __name__ == "__main__":
    # Test loading
    print("Testing data loader...")
    print("\nAvailable datasets in suite 337:")
    print(list_available_datasets(337))
    
    # Load a simple dataset
    print("\n\nLoading first dataset from suite 337...")
    X_train, X_test, y_train, y_test, cat_ind, names = load_openml_dataset(suite_id=337)
    print(f"\nFeature names: {names[:5]}...")

"""
Deep learning models for tabular data benchmark.
Includes MLP and TabNet.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MLPModel(nn.Module):
    """Simple Multi-Layer Perceptron for tabular data."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64], 
                 output_dim: int = 2, dropout: float = 0.2):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


class MLP:
    """
    MLP wrapper with sklearn-like API.
    """
    
    def __init__(self, task_type: str = 'classification', 
                 hidden_dims: list = [128, 64],
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 epochs: int = 100,
                 early_stopping_patience: int = 10,
                 device: str = None):
        
        self.task_type = task_type
        self.is_classifier = task_type == 'classification'
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  MLP using device: {self.device}")
        if self.device == 'cuda':
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler() if not self.is_classifier else None
        self.classes_ = None
        
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = False):
        """Prepare data for training/inference."""
        # Handle categorical features
        X_processed = X.copy()
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            X_processed[col] = X_processed[col].astype('category').cat.codes
        
        X_numeric = X_processed.select_dtypes(include=[np.number]).values
        
        if fit:
            X_scaled = self.scaler_X.fit_transform(X_numeric)
        else:
            X_scaled = self.scaler_X.transform(X_numeric)
        
        if y is not None:
            y_array = y.values if isinstance(y, pd.Series) else y
            if self.is_classifier:
                return torch.FloatTensor(X_scaled), torch.LongTensor(y_array)
            else:
                # For regression, also scale y
                if fit:
                    y_scaled = self.scaler_y.fit_transform(y_array.reshape(-1, 1)).flatten()
                else:
                    y_scaled = self.scaler_y.transform(y_array.reshape(-1, 1)).flatten()
                return torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled)
        
        return torch.FloatTensor(X_scaled)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train the MLP model."""
        
        # Store classes for classification
        if self.is_classifier:
            self.classes_ = np.unique(y_train)
            self.n_classes = len(self.classes_)
        
        # Prepare data
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train, fit=True)
        
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val, fit=False)
        else:
            # Use last 20% as validation
            val_size = int(0.2 * len(X_train_tensor))
            X_val_tensor = X_train_tensor[-val_size:]
            y_val_tensor = y_train_tensor[-val_size:]
            X_train_tensor = X_train_tensor[:-val_size]
            y_train_tensor = y_train_tensor[:-val_size]
        
        # Create model
        input_dim = X_train_tensor.shape[1]
        output_dim = self.n_classes if self.is_classifier else 1
        
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim, self.dropout).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        if self.is_classifier:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                if self.is_classifier:
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                X_val_tensor = X_val_tensor.to(self.device)
                y_val_tensor = y_val_tensor.to(self.device)
                val_outputs = self.model(X_val_tensor)
                
                if self.is_classifier:
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                else:
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_tensor = self._prepare_data(X_test, fit=False).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.is_classifier:
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = outputs.squeeze().cpu().numpy()
                # Inverse transform for regression
                predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classification")
        
        X_tensor = self._prepare_data(X_test, fit=False).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probabilities
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        metrics = {}
        
        if self.is_classifier:
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            if len(self.classes_) == 2:
                try:
                    proba = self.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, proba)
                except:
                    pass
        else:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
            metrics['r2'] = r2_score(y_test, predictions)
        
        return metrics


class TabNetModel:
    """
    TabNet model wrapper.
    TabNet uses sequential attention to choose which features to reason from at each decision step.
    """
    
    def __init__(self, task_type: str = 'classification',
                 n_d: int = 4, n_a: int = 4, n_steps: int = 2,
                 gamma: float = 1.3, lambda_sparse: float = 1e-4,
                 max_epochs: int = 100, patience: int = 10,
                 batch_size: int = 256):
        
        self.task_type = task_type
        self.is_classifier = task_type == 'classification'
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler() if not self.is_classifier else None
        self.classes_ = None
        
        # Check GPU availability for TabNet
        import torch
        self.tabnet_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  TabNet will use device: {self.tabnet_device}")
        if self.tabnet_device == 'cuda':
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = False):
        """Prepare data for TabNet."""
        X_processed = X.copy()
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            X_processed[col] = X_processed[col].astype('category').cat.codes

        X_numeric = X_processed.select_dtypes(include=[np.number]).values

        if fit:
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_scaled = self.scaler.transform(X_numeric)

        if y is not None:
            y_array = y.values if isinstance(y, pd.Series) else y
            if not self.is_classifier:
                if fit:
                    y_array = self.scaler_y.fit_transform(y_array.reshape(-1, 1)).flatten()
                else:
                    y_array = self.scaler_y.transform(y_array.reshape(-1, 1)).flatten()
            return X_scaled, y_array

        return X_scaled
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train TabNet model."""
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        except ImportError:
            print("TabNet not installed. Please run: pip install pytorch-tabnet")
            raise
        
        # Store classes
        if self.is_classifier:
            self.classes_ = np.unique(y_train)
        
        # Prepare data
        X_train_processed, y_train_processed = self._prepare_data(X_train, y_train, fit=True)
        
        if X_val is not None and y_val is not None:
            X_val_processed, y_val_processed = self._prepare_data(X_val, y_val, fit=False)
        else:
            # Use last 20% as validation
            val_size = int(0.2 * len(X_train_processed))
            X_val_processed = X_train_processed[-val_size:]
            y_val_processed = y_train_processed[-val_size:]
            X_train_processed = X_train_processed[:-val_size]
            y_train_processed = y_train_processed[:-val_size]
        
        # Dynamically adjust batch_size based on dataset size
        # For small datasets, use smaller batch_size to ensure enough batches per epoch
        n_samples = len(X_train_processed)
        adjusted_batch_size = min(self.batch_size, max(16, n_samples // 4))
        if adjusted_batch_size != self.batch_size:
            print(f"  Adjusting batch_size from {self.batch_size} to {adjusted_batch_size} for {n_samples} samples")
        
        # Create model
        if self.is_classifier:
            self.model = TabNetClassifier(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                lambda_sparse=self.lambda_sparse,
                verbose=0,
                seed=42
            )
            
            self.model.fit(
                X_train=X_train_processed,
                y_train=y_train_processed,
                eval_set=[(X_val_processed, y_val_processed)],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=adjusted_batch_size
            )
        else:
            self.model = TabNetRegressor(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                lambda_sparse=self.lambda_sparse,
                verbose=0,
                seed=42
            )
            
            self.model.fit(
                X_train=X_train_processed,
                y_train=y_train_processed.reshape(-1, 1),
                eval_set=[(X_val_processed, y_val_processed.reshape(-1, 1))],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=adjusted_batch_size
            )
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_processed = self._prepare_data(X_test, fit=False)
        predictions = self.model.predict(X_processed)

        if self.is_classifier:
            return predictions
        else:
            predictions = predictions.flatten()
            return self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classification")
        
        X_processed = self._prepare_data(X_test, fit=False)
        return self.model.predict_proba(X_processed)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        metrics = {}
        
        if self.is_classifier:
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            if len(self.classes_) == 2:
                try:
                    proba = self.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, proba)
                except:
                    pass
        else:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
            metrics['r2'] = r2_score(y_test, predictions)
        
        return metrics


class FTTransformerModule(nn.Module):
    """Feature Tokenizer + Transformer (Gorishniy et al., 2021)."""

    def __init__(self, n_features: int, d_token: int = 64, n_heads: int = 8,
                 n_layers: int = 3, dropout: float = 0.1, output_dim: int = 2):
        super().__init__()
        # Each feature gets its own linear projection: token_i = x_i * W_i + b_i
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=max(int(d_token * 4 / 3), n_heads),
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, output_dim)

    def forward(self, x):
        # x: (B, n_features)
        tokens = x.unsqueeze(-1) * self.W + self.b  # (B, n_features, d_token)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)     # (B, n_features+1, d_token)
        out = self.transformer(tokens)
        return self.head(self.norm(out[:, 0]))        # CLS token


class SAINTBlock(nn.Module):
    """SAINT block with row-wise and intersample (column-wise) attention."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        # Row attention: self-attention over features within each sample
        self.row_norm1 = nn.LayerNorm(d_model)
        self.row_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.row_norm2 = nn.LayerNorm(d_model)
        self.row_ff = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model)
        )
        # Intersample attention: for each feature position, attend across the batch
        self.col_norm1 = nn.LayerNorm(d_model)
        self.col_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.col_norm2 = nn.LayerNorm(d_model)
        self.col_ff = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model)
        )

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        h = self.row_norm1(x)
        attn_out, _ = self.row_attn(h, h, h)
        x = x + attn_out
        x = x + self.row_ff(self.row_norm2(x))

        # Transpose so feature-dim is treated as the batch for column attention
        x_t = x.permute(1, 0, 2)   # (seq_len, batch, d_model)
        h = self.col_norm1(x_t)
        col_out, _ = self.col_attn(h, h, h)
        x_t = x_t + col_out
        x_t = x_t + self.col_ff(self.col_norm2(x_t))
        return x_t.permute(1, 0, 2)  # (batch, seq_len, d_model)


class SAINTModule(nn.Module):
    """SAINT: Self-Attention and Intersample Attention Transformer (Somepalli et al., 2021)."""

    def __init__(self, n_features: int, d_token: int = 32, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1, output_dim: int = 2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        self.blocks = nn.ModuleList([
            SAINTBlock(d_token, n_heads, d_token * 4, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, output_dim)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.W + self.b
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, 0]))


class _BaseTransformerModel:
    """Shared sklearn-like wrapper for transformer-based tabular models."""

    def __init__(self, task_type: str, learning_rate: float, batch_size: int,
                 epochs: int, early_stopping_patience: int, device: Optional[str]):
        self.task_type = task_type
        self.is_classifier = task_type == 'classification'
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler() if not self.is_classifier else None
        self.classes_ = None

    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        raise NotImplementedError

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = False):
        X_processed = X.copy()
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            X_processed[col] = X_processed[col].astype('category').cat.codes
        X_numeric = X_processed.select_dtypes(include=[np.number]).values

        X_scaled = self.scaler_X.fit_transform(X_numeric) if fit else self.scaler_X.transform(X_numeric)

        if y is not None:
            y_array = y.values if isinstance(y, pd.Series) else y
            if self.is_classifier:
                return torch.FloatTensor(X_scaled), torch.LongTensor(y_array)
            else:
                if fit:
                    y_scaled = self.scaler_y.fit_transform(y_array.reshape(-1, 1)).flatten()
                else:
                    y_scaled = self.scaler_y.transform(y_array.reshape(-1, 1)).flatten()
                return torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled)

        return torch.FloatTensor(X_scaled)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        if self.is_classifier:
            self.classes_ = np.unique(y_train)
            self.n_classes = len(self.classes_)

        X_train_t, y_train_t = self._prepare_data(X_train, y_train, fit=True)

        if X_val is not None and y_val is not None:
            X_val_t, y_val_t = self._prepare_data(X_val, y_val, fit=False)
        else:
            val_size = int(0.2 * len(X_train_t))
            X_val_t = X_train_t[-val_size:].to(self.device)
            y_val_t = y_train_t[-val_size:].to(self.device)
            X_train_t = X_train_t[:-val_size]
            y_train_t = y_train_t[:-val_size]

        input_dim = X_train_t.shape[1]
        output_dim = self.n_classes if self.is_classifier else 1
        self.model = self._build_model(input_dim, output_dim).to(self.device)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                                  batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss() if self.is_classifier else nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y) if self.is_classifier else criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = (criterion(val_outputs, y_val_t) if self.is_classifier
                            else criterion(val_outputs.squeeze(), y_val_t)).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X_t = self._prepare_data(X_test, fit=False).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_t)
            if self.is_classifier:
                return torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds = outputs.squeeze().cpu().numpy()
                return self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classification")
        X_t = self._prepare_data(X_test, fit=False).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(X_t), dim=1).cpu().numpy()

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        predictions = self.predict(X_test)
        metrics = {}
        if self.is_classifier:
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            if len(self.classes_) == 2:
                try:
                    proba = self.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, proba)
                except:
                    pass
        else:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
            metrics['r2'] = r2_score(y_test, predictions)
        return metrics


class FTTransformer(_BaseTransformerModel):
    """
    FT-Transformer wrapper with sklearn-like API.
    Feature Tokenizer + Transformer (Gorishniy et al., NeurIPS 2021).
    """

    def __init__(self, task_type: str = 'classification',
                 d_token: int = 64, n_heads: int = 8, n_layers: int = 3,
                 dropout: float = 0.1, learning_rate: float = 0.0001,
                 batch_size: int = 256, epochs: int = 100,
                 early_stopping_patience: int = 10, device: str = None):
        super().__init__(task_type, learning_rate, batch_size, epochs, early_stopping_patience, device)
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        print(f"  FT-Transformer using device: {self.device}")
        if self.device == 'cuda':
            print(f"    GPU: {torch.cuda.get_device_name(0)}")

    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        return FTTransformerModule(input_dim, self.d_token, self.n_heads,
                                   self.n_layers, self.dropout, output_dim)


class SAINT(_BaseTransformerModel):
    """
    SAINT wrapper with sklearn-like API.
    Self-Attention and Intersample Attention Transformer (Somepalli et al., 2021).
    """

    def __init__(self, task_type: str = 'classification',
                 d_token: int = 32, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, learning_rate: float = 0.0001,
                 batch_size: int = 256, epochs: int = 100,
                 early_stopping_patience: int = 10, device: str = None):
        super().__init__(task_type, learning_rate, batch_size, epochs, early_stopping_patience, device)
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        print(f"  SAINT using device: {self.device}")
        if self.device == 'cuda':
            print(f"    GPU: {torch.cuda.get_device_name(0)}")

    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        return SAINTModule(input_dim, self.d_token, self.n_heads,
                           self.n_layers, self.dropout, output_dim)


if __name__ == "__main__":
    # Test deep learning models
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    print("Testing MLP...")
    mlp = MLP('classification', epochs=10)
    mlp.fit(X_train, y_train)
    metrics = mlp.evaluate(X_test, y_test)
    print(f"MLP Metrics: {metrics}")

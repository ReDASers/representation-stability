"""BiLSTM classifier for adversarial text detection using sensitivity features.

This module implements a bidirectional LSTM classifier with attention mechanisms
designed for detecting adversarial text samples using sequential sensitivity data.
Supports both standard 2-channel features and AWI ensemble for 4-channel features.
"""

import math
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

class SensitivityDataset(Dataset):
    """Dataset for sensitivity-based adversarial detection with multi-channel features.
    
    This dataset handles variable-length sequences of sensitivity features with optional
    positional encoding and masking for improved model performance.
    
    Attributes:
        _pos_encoding_cache: Class-level cache for positional encodings to avoid recomputation.
    """
    
    # Class-level cache for positional encodings to improve performance
    _pos_encoding_cache = {}
    
    def __init__(
        self,
        features: List[np.ndarray],
        labels: np.ndarray,
        add_mask: bool = True,
        add_position: bool = True,
        pos_encoding_type: str = 'sinusoidal',
        normalize_features: bool = True,
        num_pos_freqs: int = 4
    ):
        """Initialize the sensitivity dataset.
        
        Args:
            features: List of NumPy arrays with shape [seq_len, num_channels] containing
                sensitivity and importance scores.
            labels: Binary labels for adversarial detection (0=clean, 1=adversarial).
            add_mask: Whether to add binary mask channel (1=valid, 0=padding).
            add_position: Whether to add positional encoding channel.
            pos_encoding_type: Type of positional encoding ('linear' or 'sinusoidal').
            normalize_features: Whether to normalize features for training stability.
            num_pos_freqs: Number of frequency pairs for sinusoidal encoding.
            
        Raises:
            TypeError: If features format is not supported.
        """
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Process variable-length sequences
        if isinstance(features, list) and len(features) > 0 and isinstance(features[0], np.ndarray):
            # Determine number of base channels from first sample
            num_base_channels = features[0].shape[1] if features[0].ndim == 2 else 1
            
            max_len = max(f.shape[0] for f in features)
            batch_size = len(features)
            
            # Initialize tensors for padded features and masks
            padded_features = torch.zeros((batch_size, max_len, num_base_channels), dtype=torch.float32)
            padding_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
            
            # Fill padded tensors with actual sequence data
            for i, feat_array in enumerate(features):
                seq_len = feat_array.shape[0]
                padded_features[i, :seq_len, :] = torch.from_numpy(feat_array)
                padding_mask[i, :seq_len] = 1.0  # 1 for valid positions, 0 for padding
            
            self.is_variable_length = True
            self.seq_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
            self.raw_features = padded_features  # [batch_size, max_len, num_base_channels]
            self.padding_mask = padding_mask      # [batch_size, max_len]
            
        else:
            raise TypeError(
                f"Unsupported features type: {type(features)}. "
                f"Expected List[np.ndarray] with shape [seq_len, num_channels]."
            )
        
        # Normalize features channel-wise using only valid (non-padded) positions
        if normalize_features:
            for channel in range(num_base_channels):
                channel_data = self.raw_features[:, :, channel]
                valid_mask = (self.padding_mask == 1)
                
                if valid_mask.sum() > 0:  # Ensure we have valid positions to normalize
                    valid_values = channel_data[valid_mask]
                    mean_val = valid_values.mean()
                    std_val = valid_values.std()
                    if std_val > 0:  # Avoid division by zero
                        self.raw_features[:, :, channel][valid_mask] = (
                            channel_data[valid_mask] - mean_val
                        ) / std_val
        
        batch_size, seq_len, _ = self.raw_features.shape
        
        # Start with base sensitivity features
        feature_channels = [self.raw_features]  # [batch_size, seq_len, num_base_channels]
        
        # Optionally add padding mask as additional channel
        if add_mask:
            feature_channels.append(self.padding_mask.unsqueeze(2))  # [batch_size, seq_len, 1]
        
        # Optionally add positional encoding
        if add_position:
            if pos_encoding_type == 'linear':
                # Simple normalized linear position encoding: position/max_length
                positions = torch.arange(0, seq_len, dtype=torch.float32).repeat(batch_size, 1)
                positions = positions / seq_len  # Normalize to [0, 1] range
                positions = positions.unsqueeze(2)  # [batch_size, seq_len, 1]
                feature_channels.append(positions)
                
            elif pos_encoding_type == 'sinusoidal':
                # Transformer-style sinusoidal positional encoding with caching
                device = self.raw_features.device
                cache_key = f"{seq_len}_{num_pos_freqs}_{device}"
                
                if cache_key not in SensitivityDataset._pos_encoding_cache:
                    # Generate new positional encoding: PE(pos, 2i) = sin(pos/10000^(2i/d))
                    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
                    k = torch.arange(num_pos_freqs, dtype=torch.float32, device=device).unsqueeze(0)
                    div_term = torch.exp(-math.log(10000.0) * k / num_pos_freqs)
                    angle = pos * div_term  # Broadcasting: [seq_len, 1] * [1, freqs] -> [seq_len, freqs]
                    
                    # Create interleaved sin/cos encoding (Transformer standard)
                    pe = torch.empty(seq_len, 2 * num_pos_freqs, device=device, dtype=torch.float32)
                    pe[:, 0::2] = torch.sin(angle)  # Even indices: sine
                    pe[:, 1::2] = torch.cos(angle)  # Odd indices: cosine
                    
                    # Cache for reuse across batches with same sequence length
                    SensitivityDataset._pos_encoding_cache[cache_key] = pe
                
                # Retrieve cached encoding and expand for current batch
                pe = SensitivityDataset._pos_encoding_cache[cache_key]
                positions = pe.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, 2*freqs]
                feature_channels.append(positions)
                
            else:
                raise ValueError(f"Unsupported pos_encoding_type: {pos_encoding_type}")
        
        # Combine all feature channels into final tensor
        self.features = torch.cat(feature_channels, dim=2)  # [batch_size, seq_len, total_channels]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            For variable-length sequences: (features, label, sequence_length)
            For fixed-length sequences: (features, label)
        """
        if self.is_variable_length:
            return self.features[idx], self.labels[idx], self.seq_lengths[idx]
        else:
            return self.features[idx], self.labels[idx]

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier with attention for adversarial text detection.
    
    This model processes sequential sensitivity features using bidirectional LSTM
    layers followed by multi-head attention and pooling strategies for classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout_rate: float = 0.2,
        feature_channels: int = 3,
        use_input_projection: bool = True,
        attention_heads: int = 1,
        use_alternative_pooling: bool = False,
        num_base_channels: int = 1,
        add_mask: bool = True
    ):
        """Initialize the BiLSTM classifier.
        
        Args:
            input_dim: Sequence length (time steps).
            hidden_dim: Hidden dimension size for LSTM layers.
            num_layers: Number of LSTM layers.
            output_dim: Number of output classes (typically 2 for binary classification).
            dropout_rate: Dropout probability for regularization.
            feature_channels: Total number of input feature channels.
            use_input_projection: Whether to project input features before LSTM.
            attention_heads: Number of attention heads.
            use_alternative_pooling: Whether to use max/avg pooling with attention.
            num_base_channels: Number of base input channels (e.g., sensitivity, importance).
            add_mask: Whether mask channel was added by the dataset.
        """
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.feature_channels = feature_channels
        self.use_input_projection = use_input_projection
        self.attention_heads = attention_heads
        self.use_alternative_pooling = use_alternative_pooling
        self.num_base_channels = num_base_channels
        self.add_mask = add_mask
        
        # Optional input projection layer to transform features
        if use_input_projection:
            self.input_projection = nn.Sequential(
                nn.Linear(feature_channels, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            lstm_input_dim = hidden_dim
        else:
            lstm_input_dim = feature_channels
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Multi-head attention for sequence representation
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional LSTM output size
            num_heads=attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate final layer input dimension based on pooling strategy
        if use_alternative_pooling:
            # Concatenate attention + max + avg pooling
            fc1_input_dim = hidden_dim * 2 * 3
        else:
            # Attention pooling only
            fc1_input_dim = hidden_dim * 2
        
        # Classification head
        self.fc1 = nn.Linear(fc1_input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the BiLSTM classifier.
        
        Args:
            x: Input tensor with shape [batch_size, seq_len, feature_channels].
            seq_lengths: Optional tensor of actual sequence lengths for packed sequences.
            
        Returns:
            Classification logits with shape [batch_size, output_dim].
        """
        batch_size, seq_len, _ = x.size()
        
        # Extract padding mask from input features if available
        if self.add_mask:
            mask_channel_index = self.num_base_channels
            if x.size(2) > mask_channel_index:
                padding_mask = x[:, :, mask_channel_index].clone()
                # Convert to boolean mask (True=valid, False=padding)
                attention_mask = (padding_mask > 0.5)
            else:
                # Fallback: treat all positions as valid
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        else:
            # No mask channel: all positions are valid
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
            
        # Apply input projection if configured
        if self.use_input_projection:
            lstm_input = self.input_projection(x)
        else:
            lstm_input = x
        
        # Process sequences through LSTM (with optional packing for efficiency)
        if seq_lengths is not None:
            # Sort sequences by length for packed sequence processing
            # This improves LSTM efficiency by avoiding computation on padding
            seq_lengths_cpu = seq_lengths.cpu()
            sorted_lengths, indices = torch.sort(seq_lengths_cpu, descending=True)
            sorted_input = lstm_input[indices]
            sorted_mask = attention_mask[indices] if attention_mask is not None else None
            
            # Pack sequences: removes padding tokens from computation
            packed_input = pack_padded_sequence(sorted_input, sorted_lengths, batch_first=True)
            packed_output, _ = self.lstm(packed_input)
            
            # Unpack and restore original batch order
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
            _, reverse_indices = torch.sort(indices)
            lstm_out = lstm_out[reverse_indices]
            
            # Restore attention mask order to match original batch
            if sorted_mask is not None:
                attention_mask = sorted_mask[reverse_indices]
        else:
            # Standard LSTM forward pass (processes all positions including padding)
            lstm_out, _ = self.lstm(lstm_input)  # [batch_size, seq_len, hidden_dim*2]
        
        # Apply dropout for regularization
        lstm_out = self.dropout(lstm_out)
        
        # Prepare mask for multi-head attention (False=valid, True=padding)
        key_padding_mask = ~attention_mask
        
        # Apply self-attention to LSTM outputs
        attn_output, _ = self.mha(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            key_padding_mask=key_padding_mask
        )  # [batch_size, seq_len, hidden_dim*2]
        
        # Masked average pooling: only pool over valid (non-padding) positions
        expanded_mask = attention_mask.unsqueeze(2).expand_as(attn_output).float()
        masked_sum = torch.sum(attn_output * expanded_mask, dim=1)  # Sum valid positions only
        token_counts = attention_mask.float().sum(dim=1, keepdim=True).clamp(min=1e-9)  # Avoid div by 0
        context = masked_sum / token_counts  # Average over valid positions [batch_size, hidden_dim*2]
        
        # Additional pooling strategies if enabled
        if self.use_alternative_pooling:
            # Masked max pooling
            mask_for_max = expanded_mask.clone()
            mask_for_max[mask_for_max == 0] = -1e9  # Large negative value for masking
            max_pooled = torch.max(lstm_out + mask_for_max, dim=1)[0]
            
            # Masked average pooling of LSTM output
            lstm_masked_sum = torch.sum(lstm_out * expanded_mask, dim=1)
            avg_pooled = lstm_masked_sum / token_counts
            
            # Combine all pooling strategies
            context = torch.cat([context, max_pooled, avg_pooled], dim=1)
        
        # Final classification layers
        out = self.fc1(context)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class AWIEnsembleClassifier(nn.Module):
    """Ensemble classifier for AWI 4-channel features.
    
    Processes each of the 4 AWI channels (VG, GBP, LRP, IG) with separate BiLSTM networks
    and combines their representations for final classification. This approach follows
    the ensemble strategy while using our enhanced BiLSTM architecture.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout_rate: float = 0.2,
        use_input_projection: bool = False,
        attention_heads: int = 1,
        use_alternative_pooling: bool = False
    ):
        """Initialize the AWI ensemble classifier.
        
        Args:
            input_dim: Sequence length (time steps).
            hidden_dim: Hidden dimension size for each LSTM.
            num_layers: Number of LSTM layers per channel.
            output_dim: Number of output classes.
            dropout_rate: Dropout probability for regularization.
            use_input_projection: Whether to use input projection (unused for single channels).
            attention_heads: Number of attention heads per channel.
            use_alternative_pooling: Whether to use additional pooling strategies.
        """
        super(AWIEnsembleClassifier, self).__init__()
        
        self.num_channels = 4  # AWI channels: VG, GBP, LRP, IG
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.channel_names = ["VG", "GBP", "LRP", "IG"]
        self.use_alternative_pooling = use_alternative_pooling
        
        # Create separate BiLSTM for each AWI channel
        self.bilstms = nn.ModuleList()
        for i in range(self.num_channels):
            lstm = nn.LSTM(
                input_size=1,  # Single channel input
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0
            )
            self.bilstms.append(lstm)
            
        # Create attention layers for each channel
        self.attention_layers = nn.ModuleList()
        for i in range(self.num_channels):
            attn = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,  # Bidirectional LSTM output size
                num_heads=attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_layers.append(attn)
            
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate combined feature dimension
        if use_alternative_pooling:
            single_channel_dim = hidden_dim * 2 * 3  # Attention + max + avg pooling
        else:
            single_channel_dim = hidden_dim * 2  # Attention pooling only
            
        combined_dim = single_channel_dim * self.num_channels
        
        # Final combiner network
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for the AWI ensemble.
        
        Args:
            x: Input tensor with shape [batch_size, seq_len, num_features] containing 4 AWI channels.
            seq_lengths: Optional sequence lengths for packed sequence processing.
            
        Returns:
            Classification logits with shape [batch_size, output_dim].
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Extract AWI channels (first 4 channels)
        awi_channels = x[:, :, :4]  # [batch_size, seq_len, 4]
        
        # Create attention mask based on sequence lengths
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        if seq_lengths is not None:
            for i in range(batch_size):
                if seq_lengths[i] < seq_len:
                    attention_mask[i, seq_lengths[i]:] = False
        
        # Convert to key padding mask format for attention
        key_padding_mask = ~attention_mask
                
        # Process each AWI channel through its dedicated BiLSTM
        channel_representations = []
        
        for i in range(self.num_channels):
            # Extract single channel
            channel_data = awi_channels[:, :, i].unsqueeze(2)  # [batch_size, seq_len, 1]
            
            # Process through BiLSTM with optional sequence packing
            if seq_lengths is not None:
                # Sort for packed sequence processing
                sorted_lengths, indices = torch.sort(seq_lengths, descending=True)
                sorted_input = channel_data[indices]
                sorted_mask = key_padding_mask[indices]
                
                # Pack and process
                packed_input = pack_padded_sequence(sorted_input, sorted_lengths.cpu(), batch_first=True)
                packed_output, _ = self.bilstms[i](packed_input)
                
                # Unpack and restore order
                lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
                _, reverse_indices = torch.sort(indices)
                lstm_out = lstm_out[reverse_indices]
                key_padding_mask = sorted_mask[reverse_indices]
            else:
                # Standard forward pass
                lstm_out, _ = self.bilstms[i](channel_data)
            
            # Apply regularization and attention
            lstm_out = self.dropout(lstm_out)
            
            # Self-attention for this channel
            attn_output, _ = self.attention_layers[i](
                query=lstm_out,
                key=lstm_out,
                value=lstm_out,
                key_padding_mask=key_padding_mask
            )
            
            # Masked average pooling
            expanded_mask = attention_mask.unsqueeze(2).expand_as(attn_output).float()
            masked_sum = torch.sum(attn_output * expanded_mask, dim=1)
            token_counts = attention_mask.float().sum(dim=1, keepdim=True).clamp(min=1e-9)
            context = masked_sum / token_counts
            
            # Additional pooling strategies if enabled
            if self.use_alternative_pooling:
                # Masked max pooling
                mask_for_max = expanded_mask.clone()
                mask_for_max[mask_for_max == 0] = -1e9
                max_pooled = torch.max(lstm_out + mask_for_max, dim=1)[0]
                
                # Masked average pooling of LSTM output
                lstm_masked_sum = torch.sum(lstm_out * expanded_mask, dim=1)
                avg_pooled = lstm_masked_sum / token_counts
                
                # Combine all pooling strategies
                context = torch.cat([context, max_pooled, avg_pooled], dim=1)
            
            channel_representations.append(context)
            
        # Combine representations from all channels
        combined_features = torch.cat(channel_representations, dim=1)
        
        # Final classification
        output = self.combiner(combined_features)
        
        return output

class BiLSTMWrapper(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible BiLSTM classifier for adversarial text detection.
    
    This wrapper provides a scikit-learn interface for the BiLSTM classifier designed
    to process sequential sensitivity features. It automatically handles different feature
    types including standard 2-channel features and AWI 4-channel ensemble features.
    
    Key Features:
        - Variable-length sequence support with padding and masking
        - Automatic feature type detection and model selection
        - Early stopping with F1-based validation
        - Optional positional encoding and input projection
        - Multi-head attention with optional alternative pooling
    
    Args:
        input_dim: Maximum sequence length (auto-detected if None, capped at 128).
        hidden_dim: Hidden dimension size for LSTM layers.
        num_layers: Number of LSTM layers.
        dropout_rate: Dropout probability for regularization.
        learning_rate: Learning rate for AdamW optimizer.
        batch_size: Training batch size.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        random_state: Random seed for reproducibility.
        device: PyTorch device (auto-detected if None).
        add_mask: Whether to add binary mask channel for padding.
        add_position: Whether to add positional encoding.
        pos_encoding_type: Positional encoding type ('linear' or 'sinusoidal').
        normalize_features: Whether to normalize features channel-wise.
        use_input_projection: Whether to project input features before LSTM.
        attention_heads: Number of multi-head attention heads.
        use_alternative_pooling: Whether to use max/avg pooling with attention.
        clip_grad_norm: Gradient clipping max norm (None to disable).
        lr_scheduler: Learning rate scheduler ('onecycle', 'linear_warmup', 'cosine_warmup', None).
        num_pos_freqs: Number of frequency pairs for sinusoidal encoding.
        min_epochs: Minimum epochs before early stopping.
        min_delta: Minimum F1 improvement threshold for early stopping.
    """
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        learning_rate: float = 2e-4,
        batch_size: int = 32,
        epochs: int = 40,
        patience: int = 5,
        random_state: int = 42,
        device: Optional[torch.device] = None,
        add_mask: bool = True,
        add_position: bool = True,
        pos_encoding_type: str = 'linear',
        normalize_features: bool = True,
        use_input_projection: bool = True,
        attention_heads: int = 2,
        use_alternative_pooling: bool = True,
        clip_grad_norm: Optional[float] = None,
        lr_scheduler: Optional[str] = None,
        num_pos_freqs: int = 4,
        min_epochs: int = 5,
        min_delta: float = 1e-3
    ):

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.input_dim = input_dim
        self.add_mask = add_mask
        self.add_position = add_position
        self.pos_encoding_type = pos_encoding_type
        self.normalize_features = normalize_features
        self.use_input_projection = use_input_projection
        self.attention_heads = attention_heads
        self.use_alternative_pooling = use_alternative_pooling
        self.clip_grad_norm = clip_grad_norm
        self.lr_scheduler = lr_scheduler
        self.num_pos_freqs = num_pos_freqs
        self.min_epochs = min_epochs
        self.min_delta = min_delta

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Model state
        self.model = None
        self.is_awi_ensemble = False
        
        # External validation support
        self._no_internal_val_split = False
        self.X_val = None
        self.y_val = None
    
    def _init_model(
        self, 
        input_dim: int, 
        feature_channels: int, 
        num_base_channels: int, 
        add_mask: bool
    ) -> Union[BiLSTMClassifier, AWIEnsembleClassifier]:
        """Initialize the appropriate model based on feature type.
        
        Args:
            input_dim: Maximum sequence length.
            feature_channels: Total number of feature channels.
            num_base_channels: Number of base channels (determines model type).
            add_mask: Whether mask channel is included.
            
        Returns:
            Initialized model (BiLSTMClassifier or AWIEnsembleClassifier).
        """
        # Set random seeds for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Detect AWI ensemble mode (4 channels)
        self.is_awi_ensemble = num_base_channels == 4
        
        if self.is_awi_ensemble:
            print("Detected AWI 4-channel features. Using ensemble model with separate BiLSTM per channel.")
            model = AWIEnsembleClassifier(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=2,
                dropout_rate=self.dropout_rate,
                use_input_projection=self.use_input_projection,
                attention_heads=self.attention_heads,
                use_alternative_pooling=self.use_alternative_pooling
            )
        else:
            print(f"Using standard BiLSTM model for {num_base_channels}-channel features.")
            model = BiLSTMClassifier(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=2,
                dropout_rate=self.dropout_rate,
                feature_channels=feature_channels,
                use_input_projection=self.use_input_projection,
                attention_heads=self.attention_heads,
                use_alternative_pooling=self.use_alternative_pooling,
                num_base_channels=num_base_channels,
                add_mask=add_mask
            )
        
        return model
    
    def fit(self, X: List[np.ndarray], y: np.ndarray) -> 'BiLSTMWrapper':
        """Fit the BiLSTM classifier to training data.
        
        Args:
            X: List of numpy arrays with shape [seq_len, num_channels] containing
               sensitivity features for each sample.
            y: Binary labels (0=clean, 1=adversarial) with shape [num_samples].
            
        Returns:
            Self for method chaining.
            
        Raises:
            TypeError: If input format is not supported.
            ValueError: If input features don't have proper channel dimension.
        """
        # Validate input format
        if not (isinstance(X, list) and len(X) > 0 and isinstance(X[0], np.ndarray)):
            raise TypeError(
                f"Unsupported input type: {type(X)}. "
                f"Expected List[np.ndarray] with shape [seq_len, num_channels]."
            )
            
        if X[0].ndim < 2:
            raise ValueError(
                "Input features must have channel dimension. "
                "Shape should be [seq_len, num_channels]."
            )
            
        self.is_variable_length = True
            
        # Determine feature characteristics
        num_base_channels = X[0].shape[1]
        
        # Auto-detect sequence length with 128 token limit
        if self.input_dim is None:
            max_seq_len = max(f.shape[0] for f in X)
            self.input_dim = min(max_seq_len, 128)
            print(
                f"Auto-detected sequence length: {max_seq_len} "
                f"(using {self.input_dim}, capped at 128 tokens)"
            )
        
        # Detect and configure feature type
        if num_base_channels == 4:
            print("Detected AWI 4-channel features (VG, GBP, LRP, IG)")
            feature_type = "awi"
        elif num_base_channels == 2:
            print("Detected standard 2-channel features (sensitivity + importance)")
            feature_type = "standard"
        else:
            print(f"Detected {num_base_channels}-channel custom features")
            feature_type = "custom"
        
        # Use consistent settings across all feature types
        using_add_mask = self.add_mask
        using_add_position = self.add_position
        learning_rate = self.learning_rate
        
        self.feature_type = feature_type
        
        # Calculate total feature channels after augmentation
        total_feature_channels = num_base_channels
        if using_add_mask:
            total_feature_channels += 1
        if using_add_position:
            if self.pos_encoding_type == 'linear':
                total_feature_channels += 1
            elif self.pos_encoding_type == 'sinusoidal':
                total_feature_channels += 2 * self.num_pos_freqs
        
        print(
            f"Feature configuration - Base: {num_base_channels}, "
            f"Mask: {using_add_mask}, Position: {using_add_position} "
            f"({self.pos_encoding_type if using_add_position else 'N/A'}), "
            f"Total: {total_feature_channels}"
        )
        
        # Initialize and setup model
        self.model = self._init_model(
            input_dim=self.input_dim,
            feature_channels=total_feature_channels,
            num_base_channels=num_base_channels,
            add_mask=using_add_mask
        )
        self.model.to(self.device)
        
        # Log model complexity
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized: {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        # Setup validation data
        if self._no_internal_val_split and self.X_val is not None and self.y_val is not None:
            X_train, y_train = X, y
            X_val, y_val = self.X_val, self.y_val
            print(f"Using external validation data: {len(X_val)} samples")
        else:
            # Create validation split
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state, stratify=y
                )
                print(f"Created validation split: {len(X_val)} samples")
            except Exception as e:
                print(f"Stratified split failed ({e}), using simple split")
                # Fallback for edge cases
                val_size = int(len(X) * 0.1)
                X_train, X_val = X[val_size:], X[:val_size]
                y_train, y_val = y[val_size:], y[:val_size]
                print(f"Created simple validation split: {len(X_val)} samples")
        
        # Create datasets with consistent feature processing
        dataset_kwargs = {
            'add_mask': using_add_mask,
            'add_position': using_add_position,
            'pos_encoding_type': self.pos_encoding_type,
            'normalize_features': self.normalize_features,
            'num_pos_freqs': self.num_pos_freqs
        }
        
        train_dataset = SensitivityDataset(X_train, y_train, **dataset_kwargs)
        val_dataset = SensitivityDataset(X_val, y_val, **dataset_kwargs)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        print(f"Training setup - LR: {learning_rate}, Batch size: {self.batch_size}")
        
        # Configure learning rate scheduler
        scheduler = None
        if self.lr_scheduler == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                steps_per_epoch=len(train_loader),
                epochs=self.epochs
            )
            print("Using OneCycle learning rate scheduler")
        elif self.lr_scheduler in ['linear_warmup', 'cosine_warmup']:
            warmup_steps = int(0.05 * self.epochs * len(train_loader))
            total_steps = self.epochs * len(train_loader)
            
            if self.lr_scheduler == 'linear_warmup':
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                )
            else:  # cosine_warmup
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles=0.5
                )
            print(f"Using {self.lr_scheduler} scheduler with {warmup_steps} warmup steps")
        
        # Initialize training state
        best_val_f1 = float('-inf')
        epochs_no_improve = 0
        best_model_state = None
        training_start_time = time.time()
        
        print(f"Starting training for up to {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            epoch_start_time = time.time()
            
            for batch in train_loader:
                # Handle both regular and variable-length batches
                if len(batch) == 3:  # Variable-length sequences
                    features, labels, seq_lengths = batch
                    seq_lengths = seq_lengths.to(self.device)
                else:  # Fixed-size arrays
                    features, labels = batch
                    seq_lengths = None
                
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features, seq_lengths)
                loss = criterion(outputs, labels)
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping if configured
                if self.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Optimize
                optimizer.step()
                
                # Step learning rate scheduler if using step-based schedulers
                if scheduler is not None and (self.lr_scheduler == 'onecycle' or self.lr_scheduler == 'cosine_warmup' or self.lr_scheduler == 'linear_warmup'):
                    scheduler.step()
                
                train_loss += loss.item() * features.size(0)
            
            # Calculate average training loss and accuracy
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = correct / total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            # Store predictions for metric calculation
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Handle both regular and variable-length batches
                    if len(batch) == 3:  # Variable-length sequences
                        features, labels, seq_lengths = batch
                        seq_lengths = seq_lengths.to(self.device)
                    else:  # Fixed-size arrays
                        features, labels = batch
                        seq_lengths = None
                    
                    features, labels = features.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(features, seq_lengths)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * features.size(0)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Store predictions and labels for F1 calculation
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate average validation loss and accuracy
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total
            
            # Calculate F1 score for validation set
            # For perfectly balanced binary classification, binary average is appropriate
            val_f1 = f1_score(all_labels, all_preds, average='binary')
            val_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
            
            # Calculate metrics and timing
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - training_start_time
            
            # Create progress message
            progress_msg = (
                f"Epoch {epoch+1:2d}/{self.epochs} [{epoch_time:4.1f}s] - "
                f"Train: {train_loss:.4f}/{train_acc:.3f} | "
                f"Val: {val_loss:.4f}/{val_acc:.3f}/F1:{val_f1:.3f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Check for improvement and early stopping
            if val_f1 > best_val_f1 + self.min_delta:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                epochs_no_improve = 0
                progress_msg += f" | ★ BEST F1: {val_f1:.4f}"
            else:
                epochs_no_improve += 1
                
            progress_msg += f" ({epochs_no_improve}/{self.patience})"
            
            # Check early stopping condition
            if epochs_no_improve >= self.patience and epoch + 1 >= self.min_epochs:
                progress_msg += " - Stopping"
                print(progress_msg)
                print(f"Early stopping: Restoring best model (F1: {best_val_f1:.4f})")
                self.model.load_state_dict(best_model_state)
                break
                
            print(progress_msg)
        
        # Restore best model if available
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Training completed. Best validation F1: {best_val_f1:.4f}")
        
        return self
    
    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """Predict class labels for samples.
        
        Args:
            X: List of numpy arrays with shape [seq_len, num_channels].
            
        Returns:
            Predicted class labels with shape [num_samples].
            
        Raises:
            ValueError: If model hasn't been fitted or input format is invalid.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        if not (isinstance(X, list) and len(X) > 0 and isinstance(X[0], np.ndarray)):
            raise ValueError(
                "Input must be a list of numpy arrays with shape [seq_len, num_channels]."
            )
            
        # Apply sequence length limit
        X = [x[:128] if x.shape[0] > 128 else x for x in X]
        
        # Create dataset with same configuration as training
        dataset = SensitivityDataset(
            X, np.zeros(len(X)),  # Dummy labels
            add_mask=self.add_mask,
            add_position=self.add_position,
            pos_encoding_type=self.pos_encoding_type,
            normalize_features=self.normalize_features,
            num_pos_freqs=self.num_pos_freqs
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Generate predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:  # Variable-length sequences
                    features, _, seq_lengths = batch
                    seq_lengths = seq_lengths.to(self.device)
                else:
                    features, _ = batch
                    seq_lengths = None
                
                features = features.to(self.device)
                outputs = self.model(features, seq_lengths)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """Predict class probabilities for samples.
        
        Args:
            X: List of numpy arrays with shape [seq_len, num_channels].
            
        Returns:
            Class probabilities with shape [num_samples, 2] where column 0 is
            probability of clean text and column 1 is probability of adversarial.
            
        Raises:
            ValueError: If model hasn't been fitted or input format is invalid.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        if not (isinstance(X, list) and len(X) > 0 and isinstance(X[0], np.ndarray)):
            raise ValueError(
                "Input must be a list of numpy arrays with shape [seq_len, num_channels]."
            )
            
        # Apply sequence length limit
        X = [x[:128] if x.shape[0] > 128 else x for x in X]
        
        # Create dataset with same configuration as training
        dataset = SensitivityDataset(
            X, np.zeros(len(X)),  # Dummy labels
            add_mask=self.add_mask,
            add_position=self.add_position,
            pos_encoding_type=self.pos_encoding_type,
            normalize_features=self.normalize_features,
            num_pos_freqs=self.num_pos_freqs
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Generate probability predictions
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:  # Variable-length sequences
                    features, _, seq_lengths = batch
                    seq_lengths = seq_lengths.to(self.device)
                else:
                    features, _ = batch
                    seq_lengths = None
                
                features = features.to(self.device)
                outputs = self.model(features, seq_lengths)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        # Ensure proper 2D shape [num_samples, 2]
        probabilities = np.array(probabilities)
        if probabilities.ndim == 1:
            probabilities = np.column_stack((1 - probabilities, probabilities))
        elif probabilities.shape[1] == 1:
            probabilities = np.column_stack((1 - probabilities, probabilities))
            
        return probabilities

def create_bilstm_classifier(
    input_dim: Optional[int] = None, 
    random_state: int = 42
) -> BiLSTMWrapper:
    """Create a BiLSTM classifier with optimized default parameters.
    
    This factory function creates a BiLSTM classifier specifically designed for
    sequential sensitivity features in adversarial text detection. The model
    automatically adapts to different feature types:
    
    - Standard features (1-2 channels): Uses single BiLSTM with attention
    - AWI features (4 channels): Uses ensemble of separate BiLSTMs per channel
    
    Args:
        input_dim: Maximum sequence length (auto-detected if None, capped at 128).
        random_state: Random seed for reproducibility.
        
    Returns:
        Configured BiLSTMWrapper ready for training.
    """
    # Apply sequence length limit
    if input_dim is not None:
        input_dim = min(input_dim, 128)
    
    # Return classifier with optimized hyperparameters
    return BiLSTMWrapper(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout_rate=0.3,
        learning_rate=2e-4,
        batch_size=32,
        epochs=40,
        patience=5,
        random_state=random_state,
        add_mask=True,              # Enable padding mask
        add_position=True,          # Enable positional encoding
        pos_encoding_type='linear', # Linear positional encoding
        normalize_features=True,    # Channel-wise normalization
        use_input_projection=True,  # Input feature projection
        attention_heads=2,          # Multi-head attention
        use_alternative_pooling=True,  # Multiple pooling strategies
        clip_grad_norm=None,        # No gradient clipping
        lr_scheduler=None,          # No LR scheduling
        num_pos_freqs=4,           # Sinusoidal encoding frequencies
        min_epochs=5,              # Minimum training epochs
        min_delta=1e-3             # Early stopping threshold
    ) 
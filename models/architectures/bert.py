import torch
import torch.nn as nn
from ..base_model import BaseModel

class BERTMNIST(BaseModel):
    def __init__(self, num_classes=10, hidden_size=256, num_layers=4, num_heads=8, 
                 mlp_ratio=4.0, dropout=0.1, max_seq_length=50176):
        super(BERTMNIST, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        
        # Image embedding
        self.embedding = nn.Linear(1, hidden_size)  # 1 channel for grayscale
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, hidden_size))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),  # Ensure integer value
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Use Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
                module.out_proj.bias.data.zero_()
    
    def forward(self, x):
        # x shape: (batch_size, 1, height, width)
        batch_size = x.shape[0]
        
        # Get the actual dimensions
        height, width = x.shape[2], x.shape[3]
        seq_length = height * width
        
        # Reshape to sequence: (batch_size, seq_length, 1)
        x = x.view(batch_size, 1, -1).permute(0, 2, 1)
        
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_length, hidden_size)
        
        # Add positional encoding (only use as many positions as we have tokens)
        x = x + self.pos_embedding[:, :seq_length, :]
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, hidden_size)
        
        # Classification
        x = self.classifier(x)
        
        return x 
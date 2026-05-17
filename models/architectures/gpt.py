import torch
import torch.nn as nn
from ..base_model import BaseModel

class GPTMNIST(BaseModel):
    def __init__(self, num_classes=10, hidden_size=256, num_layers=4, num_heads=8,
                 mlp_ratio=4.0, dropout=0.1, max_seq_length=784):
        super(GPTMNIST, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        
        # Image embedding
        self.embedding = nn.Linear(1, hidden_size)  # 1 channel for grayscale
        
        # Positional encoding
        self.register_buffer('pos_embedding', torch.zeros(1, max_seq_length, hidden_size))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # GPT-style decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),  # Ensure integer value
            dropout=dropout,
            batch_first=True,
            dtype=torch.float32,  # Explicitly set dtype
            device=torch.device('cpu')  # Explicitly set device
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
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
        # x shape: (batch_size, 1, 28, 28) or (batch_size, 1, 224, 224)
        batch_size = x.shape[0]
        
        # Reshape to sequence: (batch_size, seq_len, 1)
        # For MNIST (28x28), seq_len = 784
        # For larger images (224x224), we need to resize or handle differently
        if x.shape[2] == 28 and x.shape[3] == 28:
            x = x.view(batch_size, -1, 1)  # Flatten to (batch_size, 784, 1)
        else:
            # For larger images, we need to resize to 28x28 first
            x = nn.functional.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
            x = x.view(batch_size, -1, 1)  # Flatten to (batch_size, 784, 1)
        
        # Embedding
        x = self.embedding(x)  # (batch_size, 784, hidden_size)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Create a dummy memory for the decoder
        memory = torch.zeros_like(x)
        
        # Create causal mask for autoregressive attention
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        
        # Transformer decoder
        x = self.transformer(x, memory, tgt_mask=mask)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, hidden_size)
        
        # Classification
        x = self.classifier(x)
        
        return x 
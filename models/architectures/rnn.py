import torch
import torch.nn as nn
from ..base_model import BaseModel

class LSTMMNIST(BaseModel):
    def __init__(self, num_classes=10, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=True):
        super(LSTMMNIST, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,  # MNIST is grayscale (1 channel)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate the size of the output from LSTM
        lstm_output_size = hidden_size * self.num_directions
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        batch_size = x.shape[0]
        
        # Reshape to sequence: (batch_size, 784, 1)
        x = x.view(batch_size, 1, -1).permute(0, 2, 1)
        
        # LSTM expects input of shape (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]
        
        # Classification
        x = self.classifier(last_output)
        
        return x

class GRUMNIST(BaseModel):
    def __init__(self, num_classes=10, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=True):
        super(GRUMNIST, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=1,  # MNIST is grayscale (1 channel)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate the size of the output from GRU
        gru_output_size = hidden_size * self.num_directions
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        batch_size = x.shape[0]
        
        # Reshape to sequence: (batch_size, 784, 1)
        x = x.view(batch_size, 1, -1).permute(0, 2, 1)
        
        # GRU expects input of shape (batch_size, seq_len, input_size)
        gru_out, _ = self.gru(x)
        
        # Use the last output for classification
        last_output = gru_out[:, -1, :]
        
        # Classification
        x = self.classifier(last_output)
        
        return x 
import torch.nn as nn
from .logger import ModelLogger

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = None
        
    def setup_logger(self, output_dir="outputs"):
        """Initialize the logger for this model"""
        self.logger = ModelLogger(self.get_model_name(), output_dir)
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_model_name(self):
        return self.__class__.__name__
        
    def log_model_summary(self):
        """Log model architecture and parameters"""
        if self.logger is None:
            self.setup_logger()
            
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_summary = f"""
Model: {self.get_model_name()}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Architecture:
{self}
"""
        self.logger.log_model_info(model_summary)
        
    def log_training_start(self, **kwargs):
        """Log training start with hyperparameters"""
        if self.logger is None:
            self.setup_logger()
        self.logger.log_training_start(**kwargs)
        
    def log_epoch(self, epoch, train_loss, val_loss=None, val_acc=None, **kwargs):
        """Log epoch results"""
        if self.logger is None:
            self.setup_logger()
        self.logger.log_epoch(epoch, train_loss, val_loss, val_acc, **kwargs)
        
    def log_evaluation(self, test_loss, test_acc, **kwargs):
        """Log evaluation results"""
        if self.logger is None:
            self.setup_logger()
        self.logger.log_evaluation(test_loss, test_acc, **kwargs)
        
    def log_error(self, error_msg):
        """Log error messages"""
        if self.logger is None:
            self.setup_logger()
        self.logger.log_error(error_msg)
        
    def log_warning(self, warning_msg):
        """Log warning messages"""
        if self.logger is None:
            self.setup_logger()
        self.logger.log_warning(warning_msg) 
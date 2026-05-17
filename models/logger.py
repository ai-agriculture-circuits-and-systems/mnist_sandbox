import logging
import os
from datetime import datetime

class ModelLogger:
    def __init__(self, model_name, output_dir="outputs"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.setup_logger()
        
    def setup_logger(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"{self.model_name}_{timestamp}.log")
        
        # Configure logger
        self.logger = logging.getLogger(f"{self.model_name}_{timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def log_training_start(self, **kwargs):
        self.logger.info(f"Starting training for {self.model_name}")
        for key, value in kwargs.items():
            self.logger.info(f"{key}: {value}")
            
    def log_epoch(self, epoch, train_loss, val_loss=None, val_acc=None, **kwargs):
        log_msg = f"Epoch {epoch} - Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            log_msg += f" - Val Loss: {val_loss:.4f}"
        if val_acc is not None:
            log_msg += f" - Val Acc: {val_acc:.4f}"
        for key, value in kwargs.items():
            log_msg += f" - {key}: {value:.4f}"
        self.logger.info(log_msg)
        
    def log_evaluation(self, test_loss, test_acc, **kwargs):
        self.logger.info(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
        for key, value in kwargs.items():
            self.logger.info(f"{key}: {value:.4f}")
            
    def log_model_info(self, model_summary):
        self.logger.info("Model Summary:")
        self.logger.info(model_summary)
        
    def log_error(self, error_msg):
        self.logger.error(error_msg)
        
    def log_warning(self, warning_msg):
        self.logger.warning(warning_msg) 
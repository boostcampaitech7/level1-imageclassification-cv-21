# utils/logger.py
import logging
import os
from pytorch_lightning.loggers import CSVLogger

def get_logger(save_dir, experiment_name):
    logger = CSVLogger(save_dir, name=experiment_name, version=1)
    return logger

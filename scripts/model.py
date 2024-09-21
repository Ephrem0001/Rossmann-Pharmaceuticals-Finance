import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging

# Define the log directory
log_dir = r'c:\Users\ephre\Documents\Rossmann-Pharmaceuticals-Finance-1\Logs'

# Create the log directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'model.log'),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


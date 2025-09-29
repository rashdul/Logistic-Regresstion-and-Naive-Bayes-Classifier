# Setup and imports
import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from typing import Optional, Any, Tuple, List, Dict

data = load_breast_cancer()
X_lr = data.data
y_lr = data.target

print(X_lr[1:])
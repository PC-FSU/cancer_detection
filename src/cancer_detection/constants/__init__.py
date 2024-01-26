from pathlib import Path
import torch

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import re
import json
import math
import time
import torch
import swanlab
import numpy
import random
import argparse
import contextlib
import subprocess
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names, load_from_disk

torch.manual_seed(0)

if torch.cuda.is_available:
    torch.cuda.manual_seed_all(0)
    
PG_CPU=None

from data.datasets import VQADataset

import os

import json
import tempfile
from dataclasses import asdict

from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT

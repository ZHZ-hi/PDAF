"""
Launch v3 training with proper import handling.
Bypasses the visdom dependency issue in models/__init__.py
"""
import sys
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Patch: bypass models/__init__.py which has visdom dependency
import types
fake_models = types.ModuleType('models')
fake_models.__path__ = [str(ROOT / 'models')]
sys.modules['models'] = fake_models

# Load unet_pdaf directly
spec = importlib.util.spec_from_file_location('unet_pdaf', str(ROOT / 'models/unet_pdaf.py'))
unet_pdaf = importlib.util.module_from_spec(spec)
sys.modules['models.unet_pdaf'] = unet_pdaf
spec.loader.exec_module(unet_pdaf)

# Now we can import train_unet_pdaf_v3
import train_unet_pdaf_v3 as t

# Replace the model's reference in train module
t.PDAFUNet = unet_pdaf.PDAFUNet
t.UNetBackbone = unet_pdaf.UNetBackbone

if __name__ == "__main__":
    t.main()
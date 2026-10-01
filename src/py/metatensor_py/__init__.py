# src/py/metatensor_py/__init__.py
import torch

# Importiamo l'estensione binaria locale interna al pacchetto
# from . import metatensor_core
from .metatensor import MetaTensor

# InitPattern = metatensor_core.InitPattern

__all__ = ["MetaTensor", "InitPattern"]

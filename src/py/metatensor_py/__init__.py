# src/py/metatensor_py/__init__.py

# 1. Carichiamo l'estensione binaria forzando l'importazione dal sotto-modulo annidato
from metatensor_core import metatensor_core

# 2. Carichiamo il dispatcher Python puro
from .metatensor import MetaTensor

# 3. Agganciamo l'InitPattern registrato da nanobind nel binario
InitPattern = metatensor_core.InitPattern

__all__ = ["MetaTensor", "InitPattern"]

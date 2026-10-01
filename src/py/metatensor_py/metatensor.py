# src/py/metatensor_py/metatensor.py
import torch
# Importazione dell'estensione privata annidata
from metatensor_core import metatensor_core

InitPattern = metatensor_core.InitPattern

class MetaTensor:
    """Dispatcher dinamico per varianti MetaTensor C++ fortemente tipizzate."""

    def __new__(cls, *dims, dtype="float32", pattern=InitPattern.RandomNormal, device="cpu"):
        type_map = {
            "float32": "float",
            "float": "float"
        }

        resolved_type = type_map.get(dtype)
        if not resolved_type:
            raise TypeError(f"[ERR_TYPE] Dtype non supportato dal wrapper: {dtype}")

        dim_suffix = f"_{'_'.join(str(d) for d in dims)}" if dims else ""
        cpp_class_name = f"MetaTensor_{resolved_type}{dim_suffix}"

        # Interroghiamo i metatipi estratti dall'estensione reale di nanobind
        cpp_class = getattr(metatensor_core, cpp_class_name, None)
        if cpp_class is None:
            raise KeyError(
                f"[ERR_GEOMETRY] La variante strutturata per '{cpp_class_name}' "
                f"non è stata pre-compilata nel binario OpenXLA/LibTorch."
            )

        t_device = torch.device(device)
        return cpp_class(pattern, t_device)


    @staticmethod
    def get_gradient_tape(*watched_tensors):
        """Helper opzionale per avvolgere la logica di watch se si desidera
        emulare la sintassi del GradientTape direttamente da Python.
        """
        for t in watched_tensors:
            t.watch()

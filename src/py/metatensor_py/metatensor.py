import metatensor_core
import torch

# Esponiamo l'InitPattern nativo direttamente all'utente Python
InitPattern = metatensor_core.InitPattern

class MetaTensor:
    """Dispatcher dinamico per varianti MetaTensor C++ fortemente tipizzate."""

    def __new__(cls, *dims, dtype="float32", pattern=InitPattern.RandomNormal, device="cpu"):
        # Normalizzazione dei tipi stringa accettati
        type_map = {
            "float32": "float",
            "float": "float",
            float: "float"
        }

        resolved_type = type_map.get(dtype)
        if not resolved_type:
            raise TypeError(f"[ERR_TYPE] Dtype non supportato dal wrapper: {dtype}")

        # Costruiamo il suffisso speculare al get_tensor_classname C++
        # Se dims è vuoto (es: loss scalare), il suffisso sarà vuoto
        dim_suffix = f"_{'_'.join(str(d) for d in dims)}" if dims else ""
        cpp_class_name = f"MetaTensor_{resolved_type}{dim_suffix}"

        # Interroghiamo il modulo nanobind binario
        cpp_class = getattr(metatensor_core, cpp_class_name, None)
        if cpp_class is None:
            raise KeyError(
                f"[ERR_GEOMETRY] La variante strutturata per '{cpp_class_name}' "
                f"non è stata pre-compilata nel binario OpenXLA/LibTorch."
            )

        # Convertiamo la stringa del dispositivo nel formato torch.Device digeribile da LibTorch
        t_device = torch.device(device)

        # Istanziamo ed immettiamo direttamente la classe C++ nativa
        return cpp_class(pattern, t_device)

    @staticmethod
    def get_gradient_tape(*watched_tensors):
        """Helper opzionale per avvolgere la logica di watch se si desidera
        emulare la sintassi del GradientTape direttamente da Python.
        """
        for t in watched_tensors:
            t.watch()

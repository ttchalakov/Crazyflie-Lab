"""Small multilayer perceptrons on the STM32 and, with the same bits, on the host.

``spec`` holds a network as float64 arrays (from a PyTorch ``nn.Sequential`` or its state dict), ``quant``
quantizes it to the formats of ``lib/shared/include/cf_lab/nn.h`` from calibration inputs, ``export`` writes the C
arrays an STM32 app compiles, and ``host`` builds the same C for this computer, so a host tool can run exactly what
the drone runs (``HostNet``).
"""

from cf_lab.nn.export import write_c
from cf_lab.nn.host import HostLibrary, HostNet, build_host_library, fnv1a
from cf_lab.nn.quant import FORMATS, QuantLayer, QuantNet, quantize
from cf_lab.nn.spec import Dense, Mlp, mlp_from_sequential, mlp_from_state_dict

__all__ = [
    "FORMATS",
    "Dense",
    "HostLibrary",
    "HostNet",
    "Mlp",
    "QuantLayer",
    "QuantNet",
    "build_host_library",
    "fnv1a",
    "mlp_from_sequential",
    "mlp_from_state_dict",
    "quantize",
    "write_c",
]

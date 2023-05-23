import numpy as np


def interleave_complex(a: np.ndarray, ctype: str):
    """
    Interleave real and imaginary parts of a complex-valued array (for saving it to disk).
    """
    return a if ctype == "real" else np.stack((a.real, a.imag), axis=-1)

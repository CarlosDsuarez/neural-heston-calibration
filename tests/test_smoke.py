import numpy as np
import QuantLib as ql
import torch

import nhc


def test_core_module_smoke() -> None:
    assert nhc.version() == "0.1.0"
    assert nhc.eigen_version().startswith("5.0.")

    p = nhc.HestonParams(v0=0.04, kappa=2.0, vbar=0.04, sigma=0.3, rho=-0.7)
    assert p.satisfies_feller()
    p_viol = nhc.HestonParams(v0=0.04, kappa=1.0, vbar=0.04, sigma=0.8, rho=-0.7)
    assert not p_viol.satisfies_feller()


def test_python_stack_imports() -> None:
    assert np.isfinite(1.0)
    assert torch.tensor([1.0]).item() == 1.0
    assert isinstance(ql.__version__, str)

"""Bridge Isaac Lab's Newton data arrays to plain torch tensors.

Under the Newton backend, `ArticulationData` / `RigidObjectData` / sensor data
properties return a `ProxyArray` (a warp-backed view), not a `torch.Tensor`.
Slicing one usually yields a tensor, so simple indexing code works by accident —
but passing a whole ProxyArray into a torch-script function fails at runtime with

    RuntimeError: quat_apply_inverse() Expected a value of type 'Tensor'
    for argument 'vec' but instead found type 'ProxyArray'

which only surfaces when the term is first evaluated, i.e. during environment
construction rather than at import. Wrap any whole-array access with `as_torch`.
"""

from __future__ import annotations

import torch


def as_torch(value) -> torch.Tensor:
    """Return `value` as a torch tensor, unwrapping a ProxyArray if needed."""
    torch_view = getattr(value, "torch", None)
    return torch_view if torch_view is not None else value

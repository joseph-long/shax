# SHAX: ported from jax/experimental/pallas/__init__.py (jax-v0.12.0.1); Apache-2.0, see shax/jax/_src/LICENSE-JAX.
# SHAX: hand edits are marked `SHAX:`; regenerate a diff with scripts/port-upstream.py jax/experimental/pallas/__init__.py --diff
# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for Pallas, a JAX extension for custom kernels.

See the Pallas documentation at
https://docs.jax.dev/en/latest/pallas/index.html.
"""

from shax.jax._src import core as _jax_core
from shax.jax._src.pallas.core import BlockDim as BlockDim
from shax.jax._src.pallas.core import Blocked as Blocked
from shax.jax._src.pallas.core import BlockSpec as BlockSpec
from shax.jax._src.pallas.core import BoundedSlice as BoundedSlice
from shax.jax._src.pallas.core import Buffered as Buffered
from shax.jax._src.pallas.core import CompilerParams as CompilerParams
from shax.jax._src.pallas.core import core_map as _deprecated_core_map
from shax.jax._src.pallas.core import CostEstimate as CostEstimate
from shax.jax._src.pallas.core import debug_check as debug_check
from shax.jax._src.pallas.core import Element as Element
from shax.jax._src.pallas.core import enable_debug_checks as enable_debug_checks
from shax.jax._src.pallas.core import enable_poison_buffers as enable_poison_buffers
from shax.jax._src.pallas.core import GridSpec as GridSpec
from shax.jax._src.pallas.core import Indirect as Indirect
from shax.jax._src.pallas.core import lower_as_mlir as lower_as_mlir
from shax.jax._src.pallas.core import MemoryRef as MemoryRef
from shax.jax._src.pallas.core import MemorySpace as MemorySpace
from shax.jax._src.pallas.core import no_block_spec as no_block_spec
from shax.jax._src.pallas.core import pallas_export_experimental as pallas_export_experimental
from shax.jax._src.pallas.core import RevisitMode as RevisitMode
from shax.jax._src.pallas.core import semaphore as semaphore
from shax.jax._src.pallas.core import Squeezed as Squeezed
from shax.jax._src.pallas.core import squeezed as squeezed
from shax.jax._src.pallas.cost_estimate import estimate_cost as estimate_cost
from shax.jax._src.pallas.helpers import empty as empty
from shax.jax._src.pallas.helpers import empty_like as empty_like
from shax.jax._src.pallas.helpers import empty_ref_like as empty_ref_like
from shax.jax._src.pallas.helpers import kernel as kernel
from shax.jax._src.pallas.helpers import loop as loop
from shax.jax._src.pallas.helpers import select_ref as select_ref
from shax.jax._src.pallas.helpers import when as when
from shax.jax._src.pallas.helpers import with_scoped as with_scoped
from shax.jax._src.pallas.pallas_call import pallas_call as pallas_call
from shax.jax._src.pallas.pallas_call import pallas_call_p as pallas_call_p
from shax.jax._src.pallas.primitives import debug_print as debug_print
from shax.jax._src.pallas.primitives import delay as delay
from shax.jax._src.pallas.primitives import DeviceIdType as _deprecated_DeviceIdType
from shax.jax._src.pallas.primitives import get_global as get_global
from shax.jax._src.pallas.primitives import multiple_of as multiple_of
from shax.jax._src.pallas.primitives import num_programs as num_programs
from shax.jax._src.pallas.primitives import program_id as program_id
from shax.jax._src.pallas.primitives import reciprocal as _deprecated_reciprocal
from shax.jax._src.pallas.primitives import run_scoped as run_scoped
from shax.jax._src.pallas.primitives import semaphore_read as semaphore_read
from shax.jax._src.pallas.primitives import semaphore_signal as semaphore_signal
from shax.jax._src.pallas.primitives import semaphore_wait as semaphore_wait
from shax.jax._src.pallas.utils import align_to as align_to
from shax.jax._src.pallas.utils import cdiv as cdiv
from shax.jax._src.pallas.utils import next_power_of_2 as next_power_of_2
from shax.jax._src.pallas.utils import strides_from_shape as strides_from_shape
from shax.jax._src.state.discharge import run_state as run_state
from shax.jax._src.state.indexing import ds as ds
from shax.jax._src.state.indexing import dslice as dslice
from shax.jax._src.state.indexing import Slice as Slice
from shax.jax._src.state.primitives import broadcast_to as broadcast_to


ANY = MemorySpace.ANY
HOST = _jax_core.MemorySpace.Host

_deprecations = {
    # Added September 28, 2026
    "DeviceIdType": (
        (
            "jax.experimental.pallas.DeviceIdType is deprecated and will be"
            " removed along with the device_id_type= parameter in semaphore and"
            " DMA operations."
        ),
        _deprecated_DeviceIdType,
    ),
    # Added August 17, 2026
    "reciprocal": (
        (
            "jax.experimental.pallas.reciprocal was moved to"
            " jax.experimental.pallas.tpu. Accessing it via"
            " jax.experimental.pallas is deprecated."
        ),
        _deprecated_reciprocal,
    ),
    # Added August 11, 2026
    "core_map": (
        (
            "jax.experimental.pallas.core_map is deprecated, use"
            " jax.experimental.pallas.kernel instead."
        ),
        _deprecated_core_map,
    ),
    # Finalized in JAX v0.12.0
    # TODO(slebedev): remove these for JAX v0.13.0.
    "dot": (
        (
            "pl.dot was deprecated in JAX v0.11.0, and removed in JAX v0.12.0."
            " Use jax.numpy.dot, jax.numpy.einsum or the @ operator instead in"
            " TPU or MGPU kernels."
        ),
        None,
    ),
    "debug_checks_enabled": (
        (
            "pl.debug_checks_enabled was deprecated in JAX v0.11.0, and removed"
            " in JAX v0.12.0. Use pl.enable_debug_checks instead."
        ),
        None,
    ),
}

import typing
if typing.TYPE_CHECKING:
  DeviceIdType = _deprecated_DeviceIdType
  core_map = _deprecated_core_map
  reciprocal = _deprecated_reciprocal
else:
  from shax.jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _jax_core

# SHAX: ported from jax/scipy/__init__.py (jax-v0.12.0.1); Apache-2.0, see shax/jax/_src/LICENSE-JAX.
# SHAX: hand edits are marked `SHAX:`; regenerate a diff with scripts/port-upstream.py jax/scipy/__init__.py --diff
# Copyright 2018 The JAX Authors.
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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from shax.jax.scipy import interpolate as interpolate
  from shax.jax.scipy import linalg as linalg
  from shax.jax.scipy import ndimage as ndimage
  from shax.jax.scipy import signal as signal
  from shax.jax.scipy import sparse as sparse
  from shax.jax.scipy import special as special
  from shax.jax.scipy import stats as stats
  from shax.jax.scipy import fft as fft
  from shax.jax.scipy import cluster as cluster
  from shax.jax.scipy import integrate as integrate
else:
  import shax.jax._src.lazy_loader as _lazy
  __getattr__, __dir__, __all__ = _lazy.attach(__name__, [
    "interpolate",
    "linalg",
    "ndimage",
    "signal",
    "sparse",
    "special",
    "stats",
    "fft",
    "cluster",
    "integrate",
  ])
  del _lazy

del TYPE_CHECKING

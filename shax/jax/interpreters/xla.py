# SHAX: ported from jax/interpreters/xla.py (jax-v0.12.0.1); Apache-2.0, see shax/jax/_src/LICENSE-JAX.
# SHAX: hand edits are marked `SHAX:`; regenerate a diff with scripts/port-upstream.py jax/interpreters/xla.py --diff
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

__all__ = ["apply_primitive", "canonicalize_dtype_handlers", "Backend"]

from shax.jax._src.dtypes import (
  canonicalize_value_handlers as canonicalize_dtype_handlers
)

from shax.jax._src.dispatch import (
  apply_primitive as apply_primitive,
)

from shax.jax._src.lib import _jax
Backend = _jax.Client
del _jax

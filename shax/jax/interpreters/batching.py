# SHAX: ported from jax/interpreters/batching.py (jax-v0.12.0.1); Apache-2.0, see shax/jax/_src/LICENSE-JAX.
# SHAX: hand edits are marked `SHAX:`; regenerate a diff with scripts/port-upstream.py jax/interpreters/batching.py --diff
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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from shax.jax._src.interpreters.batching import (
  axis_primitive_batchers as axis_primitive_batchers,
  batch_jaxpr_axes as batch_jaxpr_axes,
  batch_subtrace as batch_subtrace,
  bdim_at_front as bdim_at_front,
  broadcast as broadcast,
  defbroadcasting as defbroadcasting,
  defreducer as defreducer,
  defvectorized as defvectorized,
  fancy_primitive_batchers as fancy_primitive_batchers,
  primitive_batchers as primitive_batchers,
  register_vmappable as register_vmappable,
  unregister_vmappable as unregister_vmappable,
  zero_if_mapped as zero_if_mapped,
)

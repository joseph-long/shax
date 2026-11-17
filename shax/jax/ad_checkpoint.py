# SHAX: ported from jax/ad_checkpoint.py (jax-v0.12.0.1); Apache-2.0, see shax/jax/_src/LICENSE-JAX.
# SHAX: hand edits are marked `SHAX:`; regenerate a diff with scripts/port-upstream.py jax/ad_checkpoint.py --diff
# Copyright 2021 The JAX Authors.
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

from shax.jax._src.ad_checkpoint import (
  checkpoint_policies as checkpoint_policies,
  checkpoint_name as checkpoint_name,
  checkpoint_name_fwd as checkpoint_name_fwd,
  name_p as name_p,
  print_saved_residuals as print_saved_residuals,
)
from shax.jax._src.interpreters.partial_eval import (
  Recompute as Recompute,
  Saveable as Saveable,
  Offloadable as Offloadable,
)

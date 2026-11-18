# SHAX: ported from jax/experimental/roofline/__init__.py (jax-v0.12.0.1); Apache-2.0, see shax/jax/_src/LICENSE-JAX.
# SHAX: hand edits are marked `SHAX:`; regenerate a diff with scripts/port-upstream.py jax/experimental/roofline/__init__.py --diff
# Copyright 2024 The JAX Authors.
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
from shax.jax.experimental.roofline.roofline import (
  RooflineRuleContext as RooflineRuleContext,
)
from shax.jax.experimental.roofline.roofline import RooflineShape as RooflineShape
from shax.jax.experimental.roofline.roofline import RooflineResult as RooflineResult
from shax.jax.experimental.roofline.roofline import roofline as roofline
from shax.jax.experimental.roofline.roofline import register_roofline as register_roofline
from shax.jax.experimental.roofline.roofline import (
  register_standard_roofline as register_standard_roofline,
)
from shax.jax.experimental.roofline.roofline import roofline_and_grad as roofline_and_grad


import shax.jax.experimental.roofline.rooflines as rooflines

del rooflines

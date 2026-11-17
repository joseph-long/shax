# SHAX: ported from jax/scipy/stats/__init__.py (jax-v0.12.0.1); Apache-2.0, see shax/jax/_src/LICENSE-JAX.
# SHAX: hand edits are marked `SHAX:`; regenerate a diff with scripts/port-upstream.py jax/scipy/stats/__init__.py --diff
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

from shax.jax.scipy.stats import bernoulli as bernoulli
from shax.jax.scipy.stats import beta as beta
from shax.jax.scipy.stats import binom as binom
from shax.jax.scipy.stats import cauchy as cauchy
from shax.jax.scipy.stats import dirichlet as dirichlet
from shax.jax.scipy.stats import expon as expon
from shax.jax.scipy.stats import gamma as gamma
from shax.jax.scipy.stats import geom as geom
from shax.jax.scipy.stats import laplace as laplace
from shax.jax.scipy.stats import logistic as logistic
from shax.jax.scipy.stats import multinomial as multinomial
from shax.jax.scipy.stats import multivariate_normal as multivariate_normal
from shax.jax.scipy.stats import nbinom as nbinom
from shax.jax.scipy.stats import norm as norm
from shax.jax.scipy.stats import pareto as pareto
from shax.jax.scipy.stats import poisson as poisson
from shax.jax.scipy.stats import t as t
from shax.jax.scipy.stats import uniform as uniform
from shax.jax.scipy.stats import chi2 as chi2
from shax.jax.scipy.stats import betabinom as betabinom
from shax.jax.scipy.stats import gennorm as gennorm
from shax.jax.scipy.stats import truncnorm as truncnorm
from shax.jax._src.scipy.stats.kde import gaussian_kde as gaussian_kde
from shax.jax._src.scipy.stats._core import mode as mode, rankdata as rankdata, sem as sem
from shax.jax.scipy.stats import vonmises as vonmises
from shax.jax.scipy.stats import wrapcauchy as wrapcauchy
from shax.jax.scipy.stats import gumbel_r as gumbel_r
from shax.jax.scipy.stats import gumbel_l as gumbel_l

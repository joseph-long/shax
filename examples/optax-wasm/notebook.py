# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jaxlib",
#     "jax",
#     "shax",
#     "optax",
#     "numpy",
# ]
#
# [tool.uv.sources]
# shax = { path = "../../dist/shax-0.0.1-py3-none-any.whl" }
# jax = { path = "../../dist/jax-0.12.0.1+shax-py3-none-any.whl" }
# jaxlib = { path = "../../dist/jaxlib-0.12.0.1+shax-py3-none-any.whl" }
# ///
"""shax demo: fit a curve with jax.grad + optax, in the browser (marimo WASM).

`jax` here is shax (a pure-Python + NumPy JAX), installed from the shim
wheels in dist/ (packaging/build-wheels.sh). See examples/optax-wasm/README.md.
"""
import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import optax

    mo.md(
        f"""
        # Curve fitting with `jax.grad` + optax

        `jax` {jax.__version__} is **shax** (NumPy backend):
        `jax.__shax__ = {getattr(jax, "__shax__", False)}`.
        """
    )
    return jax, jnp, mo, optax


@app.cell
def _(jax, jnp):
    # Noisy samples of a cubic; we fit its four coefficients.
    true_coef = jnp.array([0.5, -1.0, -2.0, 1.5])
    key = jax.random.key(0)
    kx, kn = jax.random.split(key)
    x = jax.random.uniform(kx, (128,), minval=-1.5, maxval=1.5)
    y = jnp.polyval(true_coef, x) + 0.1 * jax.random.normal(kn, x.shape)
    return true_coef, x, y


@app.cell
def _(mo):
    lr = mo.ui.slider(0.005, 0.3, step=0.005, value=0.05, label="learning rate")
    steps = mo.ui.slider(50, 1000, step=50, value=400, label="Adam steps")
    mo.hstack([lr, steps])
    return lr, steps


@app.cell
def _(jax, jnp, lr, optax, steps, x, y):
    def loss_fn(coef):
        return jnp.mean((jnp.polyval(coef, x) - y) ** 2)

    opt = optax.adam(lr.value)

    @jax.jit
    def step(coef, state):
        loss, grad = jax.value_and_grad(loss_fn)(coef)
        updates, state = opt.update(grad, state, coef)
        return optax.apply_updates(coef, updates), state, loss

    coef = jnp.zeros(4)
    state = opt.init(coef)
    losses = []
    for _i in range(steps.value):
        coef, state, loss = step(coef, state)
        losses.append(float(loss))
    return coef, losses


@app.cell
def _(coef, losses, mo, true_coef):
    _fmt = lambda v: ", ".join(f"{float(c):+.3f}" for c in v)
    mo.md(
        f"""
        | | coefficients (x³, x², x, 1) |
        |---|---|
        | fitted | {_fmt(coef)} |
        | true | {_fmt(true_coef)} |

        Loss: {losses[0]:.4f} → **{losses[-1]:.4f}** after {len(losses)} steps.

        <span id="shax-result" data-loss="{losses[-1]:.6f}"></span>
        """
    )
    return


if __name__ == "__main__":
    app.run()

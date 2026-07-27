import time

import jax
import jax.numpy as jnp
from jax import lax


N_PROPAGATION_STEPS = 100
PROPAGATION_SCALE = 1.1


@jax.jit
def disturbance(x: jax.Array) -> jax.Array:
    # Equivalent to diag(x**2) for a length-2 vector.
    return jnp.eye(x.shape[0], dtype=x.dtype) @ (x**2)


@jax.jit
def propagation(E: jax.Array) -> jax.Array:
    A = PROPAGATION_SCALE * jnp.eye(E.shape[0], dtype=E.dtype)

    def scan_step(current_E, _):
        next_E = A @ current_E
        return next_E, None

    final_E, _ = lax.scan(
        scan_step,
        init=E,
        xs=None,
        length=N_PROPAGATION_STEPS,
    )
    return final_E


@jax.jit
def objective(x: jax.Array) -> jax.Array:
    E = disturbance(x)
    Phi = propagation(E)

    # Row-wise norms, followed by a sum, so the output is scalar.
    return jnp.sum(jnp.linalg.norm(Phi, axis=-1))


# Compile the objective and its gradient together.
objective_and_grad = jax.jit(jax.value_and_grad(objective))


def benchmark(
    function,
    x: jax.Array,
    num_warmup: int = 10,
    num_runs: int = 10_000,
) -> tuple[float, float]:
    """Return total runtime and average runtime in microseconds."""

    # Trigger compilation and warm up the accelerator.
    for _ in range(num_warmup):
        result = function(x)
        jax.tree.map(
            lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
            result,
        )

    start = time.perf_counter()

    for _ in range(num_runs):
        result = function(x)

    # JAX dispatch is asynchronous, so synchronize before stopping the timer.
    jax.tree.map(
        lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
        result,
    )

    elapsed_seconds = time.perf_counter() - start
    average_microseconds = elapsed_seconds / num_runs * 1e6

    return elapsed_seconds, average_microseconds


def main():
    # .T does nothing for a one-dimensional JAX array.
    x = jnp.array([3.0, 1.0], dtype=jnp.float32)

    value, dh_dx = objective_and_grad(x)

    # Synchronize before printing.
    value.block_until_ready()
    dh_dx.block_until_ready()

    print("x:")
    print(x)

    print("\nobjective(x):")
    print(value)

    print("\ndh/dx:")
    print(dh_dx)

    print("\nE(x):")
    print(disturbance(x))

    print("\nPhi:")
    print(propagation(disturbance(x)))

    total_time, average_time_us = benchmark(
        objective_and_grad,
        x,
        num_warmup=10,
        num_runs=10_000,
    )

    print("\nRuntime benchmark")
    print(f"Total runtime:   {total_time:.6f} seconds")
    print(f"Average runtime: {average_time_us:.3f} microseconds per call")


if __name__ == "__main__":
    main()
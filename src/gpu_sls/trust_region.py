"""Scale-aware adaptive trust-region utilities for the outer SQP loop."""

import jax.numpy as jnp


def scaled_step_norm(dX, dU, X, U):
    """Return a dimensionless infinity norm for an SQP primal step.

    Each state/control component is scaled by the largest magnitude of the
    corresponding current trajectory component (with a floor of one). This
    keeps a single radius meaningful for differently-scaled variables.
    """
    x_scale = jnp.maximum(1.0, jnp.max(jnp.abs(X), axis=0))
    u_scale = jnp.maximum(1.0, jnp.max(jnp.abs(U), axis=0))
    return jnp.maximum(
        jnp.max(jnp.abs(dX) / x_scale),
        jnp.max(jnp.abs(dU) / u_scale),
    )


def project_direction(dX, dU, dV, X, U, radius):
    """Radially project a search direction into the scaled trust region."""
    norm = scaled_step_norm(dX, dU, X, U)
    scale = jnp.minimum(
        1.0,
        radius / jnp.maximum(norm, jnp.finfo(dX.dtype).eps),
    )
    return scale * dX, scale * dU, scale * dV, norm, scale


def filter_reduction_ratios(
    current_cost,
    current_residual,
    trial_cost,
    trial_residual,
    model_cost_change,
    model_residual_change,
):
    """Return separate objective and feasibility agreement ratios.

    GPU-SLS globalizes SQP with a filter: a step may be useful because it
    improves either the objective or feasibility. Collapsing those quantities
    into a fixed-weight merit can reject a step already accepted by that
    filter, and shrinking the same direction cannot resolve the disagreement.
    These two ratios preserve the filter semantics while still measuring how
    accurately the local model predicted each improvement.
    """
    current_theta = 0.5 * jnp.vdot(current_residual, current_residual)
    trial_theta = 0.5 * jnp.vdot(trial_residual, trial_residual)
    model_residual = current_residual + model_residual_change
    model_theta = 0.5 * jnp.vdot(model_residual, model_residual)

    actual_cost_reduction = current_cost - trial_cost
    predicted_cost_reduction = -model_cost_change
    actual_feas_reduction = current_theta - trial_theta
    predicted_feas_reduction = current_theta - model_theta

    eps = jnp.finfo(current_cost.dtype).eps
    valid_cost = predicted_cost_reduction > eps * jnp.maximum(
        1.0, jnp.abs(current_cost)
    )
    valid_feas = predicted_feas_reduction > eps * jnp.maximum(
        1.0, jnp.abs(current_theta)
    )
    cost_ratio = jnp.where(
        valid_cost,
        actual_cost_reduction / predicted_cost_reduction,
        -jnp.inf,
    )
    feas_ratio = jnp.where(
        valid_feas,
        actual_feas_reduction / predicted_feas_reduction,
        -jnp.inf,
    )
    return (
        cost_ratio,
        feas_ratio,
        actual_cost_reduction,
        predicted_cost_reduction,
        actual_feas_reduction,
        predicted_feas_reduction,
        valid_cost,
        valid_feas,
    )


def update_radius(
    radius,
    ratio,
    step_norm,
    accepted,
    min_radius,
    max_radius,
    shrink_threshold,
    expand_threshold,
    shrink_factor,
    expand_factor,
    boundary_fraction,
):
    """Update a trust-region radius using standard agreement thresholds."""
    shrink = jnp.logical_or(jnp.logical_not(accepted), ratio < shrink_threshold)
    boundary_step = step_norm >= boundary_fraction * radius
    expand = accepted & (ratio > expand_threshold) & boundary_step

    new_radius = jnp.where(shrink, shrink_factor * radius, radius)
    new_radius = jnp.where(expand, expand_factor * radius, new_radius)
    return jnp.clip(new_radius, min_radius, max_radius)

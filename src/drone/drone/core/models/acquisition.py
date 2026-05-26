import numpy as np


def ucb(
    X,
    gp_model,
    beta=2.0
):
    # Get predictions from the GP model
    mean, std = gp_model.predict(X)
    ucb_values = mean + np.sqrt(beta) * std

    return ucb_values


def ucb_beta(t, d):
    # Calculate the information gain proxy
    gamma_t = 2 * np.log(t + 1)
    # Theoretical beta value
    return 2 * (d**2) + 300 * gamma_t * (np.log(t + 1) ** 3)


def select_ucb_action(
    action_space,
    context,
    gp_model,
    t,
    d=None,
    safe_set=None
):
    if d is None:
        d = action_space.shape[1] + context.shape[0]

    # Calculate beta parameter
    beta_t = ucb_beta(t, d)

    # If safe_set is not provided, all actions are considered safe
    if safe_set is None:
        safe_set = action_space

    # Prepare inputs for GP model by combining actions with context
    # Create an array where each action is paired with the current context
    inputs = np.array([np.concatenate([action, context])
                      for action in safe_set])

    # Calculate UCB values for all action-context pairs
    ucb_values = ucb(inputs, gp_model, beta=beta_t)

    # Select the action with the highest UCB value
    best_idx = np.argmax(ucb_values)
    best_action = safe_set[best_idx]
    best_ucb = ucb_values[best_idx]

    return best_action, best_ucb

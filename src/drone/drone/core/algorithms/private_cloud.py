import numpy as np
import logging
import sys
sys.path.append("<src_dir>")

from drone.core.models import DroneGaussianProcess, select_ucb_action
from ksurf import Ksurf

logger = logging.getLogger(__name__)


class PrivateCloudBandit:
    def __init__(
        self,
        action_space,
        resource_limit,
        initial_safe_set=None,
        exploration_duration=10,
        confidence_level=0.1,
        sliding_window_size=30,
        gp_hyperparams=None
    ):
        self.action_space = action_space
        self.resource_limit = resource_limit
        self.exploration_duration = exploration_duration
        self.confidence_level = confidence_level
        self.t = 1  # Time step counter
        self.exploration_phase = True

        # Initialize GP models
        gp_params = gp_hyperparams or {}
        self.performance_gp = DroneGaussianProcess(
            sliding_window_size=sliding_window_size,
            **gp_params
        )
        self.resource_gp = DroneGaussianProcess(
            sliding_window_size=sliding_window_size,
            **gp_params
        )
        # Initialize Ksurf
        kf_params = {}
        self.performance_kf = Ksurf(
            sliding_window_size=sliding_window_size,
            **kf_params
        )
        self.resource_kf = Ksurf(
            sliding_window_size=sliding_window_size,
            **kf_params
        )

        # If no initial safe set is provided, use a conservative approach
        if initial_safe_set is None:
            safe_size = max(1, int(len(action_space) * 0.25))
            self.safe_set = action_space[:safe_size].copy()
        else:
            self.safe_set = initial_safe_set.copy()

        # History
        self.history = {
            'actions': [],
            'contexts': [],
            'performance': [],
            'resource_usage': [],
            'safe_set_size': []
        }

    def get_safe_set(
        self,
        context,
        beta_t=None
    ):
        if self.t <= self.exploration_duration:
            return self.safe_set

        if beta_t is None:
            # Calculate the dimension of the joint action-context space
            d = self.action_space.shape[1] + context.shape[0]
            # Calculate beta parameter based on theory
            beta_t = 2 * (d**2) + 300 * 2 * np.log(self.t + 1) * \
                (np.log(self.t + 1) ** 3)

        # Prepare inputs for GP model by combining actions with context
        inputs = np.array([np.concatenate([action, context])
                          for action in self.action_space])

        # Get resource usage predictions from GP model
        #mean, std = self.resource_gp.predict(inputs)
        mean, std = self.resource_kf.predict(inputs)

        # Calculate lower confidence bound on resource usage
        lcb_values = mean - np.sqrt(beta_t) * std

        # Find actions where the lower confidence bound is below the resource limit
        safe_indices = np.where(lcb_values <= self.resource_limit)[0]

        # If no safe actions are found, return the current safe set
        if len(safe_indices) == 0:
            logger.warning("No safe actions found. Using current safe set.")
            return self.safe_set

        # Update safe set
        self.safe_set = self.action_space[safe_indices]

        return self.safe_set

    def select_exploration_action(
        self,
        context
    ):
        # Randomly select an action from the safe set
        return self.safe_set[np.random.randint(len(self.safe_set))]

    def select_action(
        self,
        context
    ):
        # During the exploration phase, select random actions from safe set
        if self.t <= self.exploration_duration:
            self.exploration_phase = True
            return self.select_exploration_action(context)

        self.exploration_phase = False

        # Get the current safe set
        safe_set = self.get_safe_set(context)

        # Calculate the dimension of the joint action-context space
        d = self.action_space.shape[1] + context.shape[0]

        # Use UCB strategy to select action from safe set
        action, _ = select_ucb_action(
            action_space=self.action_space,
            context=context,
            #gp_model=self.performance_gp,
            gp_model=self.performance_kf,
            t=self.t,
            d=d,
            safe_set=safe_set
        )

        return action

    def update(
        self,
        action,
        context,
        performance,
        resource_usage
    ):
        # Check if resource limit was violated
        is_safe = resource_usage <= self.resource_limit

        # Combine action and context for GP input
        X = np.array([np.concatenate([action, context])])

        # Update performance GP model
        #self.performance_gp.update(X, np.array([performance]))
        self.performance_kf.update(X, np.array([performance]))

        # Update resource usage GP model
        #self.resource_gp.update(X, np.array([resource_usage]))
        self.resource_kf.update(X, np.array([resource_usage]))

        # Update history
        self.history['actions'].append(action)
        self.history['contexts'].append(context)
        self.history['performance'].append(performance)
        self.history['resource_usage'].append(resource_usage)
        self.history['safe_set_size'].append(len(self.safe_set))

        # Increment time step
        self.t += 1

        return performance, is_safe

    def reset(self):
        #self.performance_gp.reset()
        self.performance_kf.reset()
        #self.resource_gp.reset()
        self.resource_kf.reset()
        self.t = 1
        self.exploration_phase = True
        self.history = {
            'actions': [],
            'contexts': [],
            'performance': [],
            'resource_usage': [],
            'safe_set_size': []
        }

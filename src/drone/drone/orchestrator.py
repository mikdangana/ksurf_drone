import logging
import time
import numpy as np
import os
import yaml

from drone.core.algorithms import PublicCloudBandit, PrivateCloudBandit
from drone.utils import (
    MonitoringInterface, PrometheusMonitoring,
    ApplicationIdentifier,
    ObjectiveEnforcer, ResourceEnforcer
)
from drone.kubernetes import KubernetesClient

logger = logging.getLogger(__name__)


class DroneOrchestrator:
    def __init__(
        self,
        app_name,
        namespace="default",
        mode="public",  # "public" or "private"
        prometheus_url="http://localhost:9090",
        in_cluster=False,
        config_file=None
    ):
        self.app_name = app_name
        self.namespace = namespace
        self.mode = mode
        self.running = False
        self.iteration = 0

        # Load configuration if provided
        self.config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)

        # Initialize components
        self.k8s_client = KubernetesClient(
            namespace=namespace,
            in_cluster=in_cluster
        )
        self.app_identifier = ApplicationIdentifier(self.k8s_client)
        self.monitoring = PrometheusMonitoring(
            prometheus_url=prometheus_url,
            app_name=app_name,
            namespace=namespace
        )

        # Initialize enforcers
        if mode == "public":
            # For public cloud, use the objective enforcer
            alpha = self.config.get("alpha", 0.5)
            beta = self.config.get("beta", 0.5)
            self.enforcer = ObjectiveEnforcer(alpha=alpha, beta=beta)
        else:
            # For private cloud, use the resource enforcer
            resource_limits = self.config.get("resource_limits", None)
            self.enforcer = ResourceEnforcer(
                resource_limits=resource_limits,
                k8s_client=self.k8s_client
            )

        # Define action space
        self.build_action_space()

        # Initialize the appropriate bandit algorithm
        if mode == "public":
            alpha, beta = self.enforcer.get_weights()
            self.algorithm = PublicCloudBandit(
                action_space=self.action_space,
                alpha=alpha,
                beta=beta
            )
        else:
            # For private cloud, get the resource limit (P_max)
            # We'll use memory as the primary constraint
            resource_limits = self.enforcer.get_absolute_limits()
            p_max = resource_limits.get("memory", 0.7)

            # Get an initial safe set (conservative)
            safe_size = max(1, int(len(self.action_space) * 0.1))
            initial_safe_set = self.action_space[:safe_size]

            self.algorithm = PrivateCloudBandit(
                action_space=self.action_space,
                resource_limit=p_max,
                initial_safe_set=initial_safe_set
            )

    def build_action_space(self):
        # Get nodes to determine available zones
        nodes = self.k8s_client.get_nodes()
        zone_labels = {}

        for node in nodes:
            if "labels" in node and "zone" in node["labels"]:
                zone = node["labels"]["zone"]
                if zone not in zone_labels:
                    zone_labels[zone] = []
                zone_labels[zone].append(node["name"])

        # If no zones are found, create a default one
        if not zone_labels:
            zone_labels = {"zone-1": [node["name"] for node in nodes]}

        self.zones = zone_labels
        num_zones = len(zone_labels)

        # Define ranges for each action dimension
        cpu_values = np.linspace(0.1, 4.0, 10)
        memory_values = np.array(
            [128, 256, 512, 1024, 2048, 4096, 8192])  # Memory in Mi
        replica_values = np.array([1, 2, 3, 4, 5])  # Possible replica counts

        scheduling_values = np.array([0, 1, 2])

        # Build action space by sampling combinations
        action_space = []
        num_actions = 100  # Limit the action space size for computational efficiency

        for _ in range(num_actions):
            cpu = np.random.choice(cpu_values)
            memory = np.random.choice(memory_values)
            replicas = np.random.choice(replica_values)

            # Ensure we don't exceed replicas across zones
            scheduling = np.random.choice(scheduling_values, size=num_zones)
            if sum(scheduling) > replicas:
                # Adjust scheduling to match replica count
                while sum(scheduling) > replicas:
                    idx = np.random.randint(num_zones)
                    if scheduling[idx] > 0:
                        scheduling[idx] -= 1

            # Combine into a single action vector
            action = np.concatenate([
                [cpu],
                [memory],
                [replicas],
                scheduling
            ])

            action_space.append(action)

        self.action_space = np.array(action_space)
        logger.info(
            f"Built action space with {len(self.action_space)} actions and {self.action_space.shape[1]} dimensions")

    def action_to_parameters(self, action):
        # Extract dimensions from the action vector
        num_zones = len(self.zones)
        cpu = action[0]
        memory = action[1]
        replicas = int(action[2])
        scheduling = action[3:3+num_zones]

        # Convert memory from Mi to a Kubernetes-friendly string
        memory_str = f"{int(memory)}Mi"

        # Build node affinities based on scheduling
        node_affinities = {}

        zone_names = list(self.zones.keys())
        for i, count in enumerate(scheduling):
            if count > 0:
                zone = zone_names[i]
                node_affinities[zone] = self.zones[zone]

        return {
            "cpu": cpu,
            "memory": memory_str,
            "replicas": replicas,
            "node_affinities": node_affinities
        }

    def parameters_to_action(self, params):
        # Extract parameters
        cpu = params.get("cpu", 0.5)

        # Convert memory string to numeric value
        memory_str = params.get("memory", "512Mi")
        try:
            if memory_str.endswith('Mi'):
                memory = float(memory_str[:-2])
            elif memory_str.endswith('Gi'):
                memory = float(memory_str[:-2]) * 1024
            else:
                memory = float(memory_str)
        except (ValueError, AttributeError):
            memory = 512

        replicas = params.get("replicas", 1)
        node_affinities = params.get("node_affinities", {})

        # Build scheduling vector
        num_zones = len(self.zones)
        scheduling = np.zeros(num_zones)

        zone_names = list(self.zones.keys())
        for i, zone in enumerate(zone_names):
            if zone in node_affinities:
                # Simplified: just indicate if we're using this zone
                scheduling[i] = 1

        # Combine into a single action vector
        action = np.concatenate([
            [cpu],
            [memory],
            [replicas],
            scheduling
        ])

        return action

    def get_context(self):
        # Get context from monitoring module
        context_dict = self.monitoring.get_context()

        # Convert to a numeric vector
        context = np.array([
            context_dict.get("workload", 0.0),
            context_dict.get("cpu_util", 0.0),
            context_dict.get("mem_util", 0.0),
            context_dict.get("net_util", 0.0)
        ])

        # Add spot price if in public cloud mode
        if self.mode == "public" and "spot_price" in context_dict:
            context = np.append(context, context_dict["spot_price"])

        return context

    def calculate_cost(self, action, context):
        # Extract components
        cpu = action[0]
        memory = action[1]  # in Mi
        replicas = int(action[2])

        # Calculate base cost
        cpu_cost = cpu * 0.0425
        memory_cost = (memory / 1024) * 0.00575

        # Calculate total cost
        cost = (cpu_cost + memory_cost) * replicas

        # Adjust based on spot price if available
        if self.mode == "public" and len(context) >= 5:
            spot_price = context[4]
            # Adjust cost based on spot price
            cost = cost * spot_price

        return cost

    def orchestrate_once(self):
        self.iteration += 1
        logger.info(f"Starting orchestration iteration {self.iteration}")

        # Get the current context
        context = self.get_context()
        logger.debug(f"Current context: {context}")

        # Get the current action
        if self.iteration == 1:
            # For the first iteration, get the current resource configuration
            current_resources = self.k8s_client.get_current_resources(
                self.app_name)
            if current_resources:
                # Convert to action vector
                action = self.parameters_to_action(current_resources)
                logger.info(
                    f"Using current configuration for first iteration: {current_resources}")
            else:
                # If no current configuration, select a new action
                action = self.algorithm.select_action(context)
                logger.info(
                    "No current configuration found, selecting new action")
        else:
            # Select action based on the algorithm
            action = self.algorithm.select_action(context)

        # Convert action to resource parameters
        params = self.action_to_parameters(action)
        logger.info(f"Selected resource parameters: {params}")

        # Apply the action
        success = self.k8s_client.apply_resource_action(
            app_name=self.app_name,
            cpu=params["cpu"],
            memory=params["memory"],
            replicas=params["replicas"],
            node_affinities=params["node_affinities"]
        )

        if not success:
            logger.warning("Failed to apply resource action")

        # Wait for the action to take effect
        time.sleep(30)  # Grace period for resource changes to apply

        # Measure performance and resource usage
        perf_metrics = self.monitoring.get_performance_metrics()
        resource_usage = self.monitoring.get_resource_usage()

        # Extract performance metric (e.g., latency for microservices, job time for batch jobs)
        app_type = self.app_identifier.identify_app_type(self.app_name)
        if app_type == "microservice":
            performance = perf_metrics.get("p90_latency", 0.0)
            # For latency, lower is better, so negate the value
            performance = -performance
        else:
            performance = perf_metrics.get("job_time", 0.0)
            # For job time, lower is better, so negate the value
            performance = -performance

        # Calculate cost
        cost = self.calculate_cost(action, context)

        # Update the algorithm
        if self.mode == "public":
            # Update public cloud algorithm
            reward = self.algorithm.update(action, context, performance, cost)
            is_safe = True  # Public cloud assumes unlimited resources
        else:
            # Update private cloud algorithm
            resource_value = resource_usage.get("memory", 0.0)
            performance, is_safe = self.algorithm.update(
                action, context, performance, resource_value)
            reward = performance  # In private cloud, reward is just performance

        # Return results
        return {
            "iteration": self.iteration,
            "action": action,
            "params": params,
            "context": context,
            "performance": performance,
            "cost": cost,
            "reward": reward,
            "is_safe": is_safe
        }

    def start(self, iterations=None, interval=60):
        self.running = True
        self.iteration = 0

        logger.info(
            f"Starting Drone Orchestrator for {self.app_name} in {self.mode} mode")

        try:
            while self.running:
                # Run one iteration
                result = self.orchestrate_once()

                # Check if we should stop
                if iterations is not None and self.iteration >= iterations:
                    logger.info(f"Completed {iterations} iterations, stopping")
                    self.running = False
                    break

                # Wait for the next interval
                if self.running:
                    logger.info(
                        f"Waiting {interval} seconds until next iteration")
                    time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Orchestration interrupted by user")
            self.running = False
        except Exception as e:
            logger.error(f"Error in orchestration: {e}")
            self.running = False
        finally:
            logger.info("Drone Orchestrator stopped")

    def stop(self):
        logger.info("Stopping Drone Orchestrator")
        self.running = False

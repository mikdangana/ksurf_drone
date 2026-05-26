import argparse
import logging
import sys
from drone import DroneOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("drone")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drone Resource Orchestration Framework"
    )

    parser.add_argument(
        "--app-name",
        required=True,
        help="Name of the application to orchestrate"
    )

    parser.add_argument(
        "--namespace",
        default="default",
        help="Kubernetes namespace"
    )

    parser.add_argument(
        "--mode",
        choices=["public", "private"],
        default="public",
        help="Orchestration mode (public or private cloud)"
    )

    parser.add_argument(
        "--prometheus-url",
        default="http://localhost:9090",
        help="URL of the Prometheus server"
    )

    parser.add_argument(
        "--in-cluster",
        action="store_true",
        help="Use in-cluster Kubernetes configuration"
    )

    parser.add_argument(
        "--config-file",
        help="Path to a configuration file"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of iterations to run (default: run continuously)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval between iterations in seconds"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create and start the orchestrator
    try:
        orchestrator = DroneOrchestrator(
            app_name=args.app_name,
            namespace=args.namespace,
            mode=args.mode,
            prometheus_url=args.prometheus_url,
            in_cluster=args.in_cluster,
            config_file=args.config_file
        )

        # Start the orchestrator
        orchestrator.start(
            iterations=args.iterations,
            interval=args.interval
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

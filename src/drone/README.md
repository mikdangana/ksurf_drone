# Drone Framework

Drone is a dynamic resource orchestration framework for containerized clouds. It uses machine learning techniques (Gaussian Processes and Contextual Bandits) to adaptively configure resources to enhance application performance and reduce operational costs amidst cloud uncertainties.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Create a configuration file (`config.yaml`):

```yaml
# See config.yaml for an example configuration
```

2. Run the orchestrator:

```bash
python main.py --app-name my-application --mode public --config-file config.yaml
```

## Usage

### Command Line Options

```
--app-name       Application name to orchestrate (required)
--namespace      Kubernetes namespace (default: "default")
--mode           Orchestration mode: "public" or "private" (default: "public")
--prometheus-url URL for Prometheus server (default: http://prometheus-server.monitoring:9090)
--in-cluster     Use in-cluster Kubernetes configuration
--mock           Use mock components for testing
--config-file    Path to configuration file
--iterations     Number of orchestration iterations to run
--interval       Interval between iterations in seconds (default: 60)
--verbose        Enable verbose logging
```

# Ksurf & Drone for Cloud Resource Allocation under Highly Variable Cloud Workloads

Papers on [Ksurf-Drone](https://ieeexplore.ieee.org/abstract/document/11347600/) & [Ksurf+](https://ieeexplore.ieee.org/abstract/document/11150546/)

## Overview

This project combines **VarBench** (a variable cloud workload benchmark) with **Drone** (a dynamic resource orchestration framework) to evaluate cloud autoscaling algorithms under highly variable workloads.

There are three main operations:

1. **Deploy and launch VarBench** -- set up the benchmark environment
2. **Configure and launch Drone** -- run the autoscaler with a chosen algorithm
3. **Collect and analyze data** -- gather results from VarBench

## Operation 1: Deploy and Launch VarBench

Deploy the VarBench benchmark suite to your cloud environment.

Follow the deployment and launch instructions at [github.com/msrg/ksurf](https://github.com/msrg/ksurf).

## Operation 2: Configure and Launch Drone

Once VarBench is running, configure and launch Drone on the VarBench autoscaler node. Choose one of two private algorithm modes depending on your experiment:

- **Default private algorithm** -- Drone's built-in private cloud autoscaling
- **Ksurf private algorithm** -- Ksurf-enhanced autoscaling (KF-PCA based)

### Setup

Update `src_dir` in the private cloud algorithm code and set Tracker properties:
```
sed -i /<src_dir>/\/ksurf\/src\// drone/drone/core/algorithms/private_cloud.py
vim config/testbed.properties
```

Install dependencies if not already installed:
```
pip install -r requirements.txt
```

### Launch Drone

```bash
python main.py --app-name my-application --mode private --config-file config.yaml
```

Use the version of Drone included in this repository (`src/drone/`). For the full set of command line options and configuration details, see:
- [src/drone/README.md](src/drone/README.md)
- [github.com/msrg/drone](https://github.com/msrg/drone)

## Operation 3: Collect and Analyze VarBench Data

After experiments complete, collect benchmark data and run analysis.

Follow the data collection and analysis instructions at [github.com/msrg/ksurf](https://github.com/msrg/ksurf).

### Result data
```
ls data_tracker.pickle.csv*
```

### Workload data
```
ls data/
```

### Analyze VarBench data

Use the VarBench analysis tools to process experiment results and generate plots:
```bash
python src/varbench/varbench_analysis.py
```

For the full analysis pipeline (including metric aggregation, comparison across autoscaler configurations, and visualization), follow the instructions at [github.com/msrg/ksurf](https://github.com/msrg/ksurf).

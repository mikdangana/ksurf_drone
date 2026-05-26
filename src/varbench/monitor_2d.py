import subprocess
import time

# History tracking
hist_cpu = []
hist_mem = []
histlen = 80
maxlen = 15  # Maximum history length

# Append to history log
def writeline(log, line):
    for i in range(maxlen):
        if i >= len(log):
            log.append("")
        log[i] = log[i] + line[i] if i < len(line) else log[i] + " "

# Convert values to text bars
def to_string(v, delim="."):
    return delim * v

# Plot function for two pods
def plot(label, pod1_name, pod2_name, pod1_values, pod2_values):
    print(f"\n{label} Usage History (Last {maxlen} points)")
    print(f"{pod1_name:<20} | {pod2_name:<20}")
    print("-" * 45)

    # Plot historical usage
    for v1, v2 in zip(pod1_values, pod2_values):
        hist_cpu.append(v1)
        hist_mem.append(v2)

    max_cpu, max_mem = max(hist_cpu[-histlen:]), max(hist_mem[-histlen:])

    log_cpu, log_mem = [], []
    for v1, v2 in zip(hist_cpu[-histlen:], hist_mem[-histlen:]):
        writeline(log_cpu if label == "CPU (milliCPU)" else log_mem, to_string(v1*10/max_cpu, "."))
        writeline(log_cpu if label == "CPU (milliCPU)" else log_mem, to_string(v2*10/max_mem, "x"))

    # Print the latest log entries
    for i in range(len(log_cpu if label == "CPU (milliCPU)" else log_mem)):
        print(
            (log_cpu if label == "CPU (milliCPU)" else log_mem)[
                len(log_cpu if label == "CPU (milliCPU)" else log_mem) - i - 1
            ]
        )
    print("-" * maxlen)

# Monitoring Loop
while True:
    try:
        # Fetch pod metrics
        result = subprocess.run(
            ["kubectl", "top", "pods", "-n", "kube-system", "--no-headers"],
            stdout=subprocess.PIPE,
            text=True,
        )

        if not result.stdout.strip():
            print("No metrics available. Retrying...")
            time.sleep(5)
            continue

        # Parse metrics and sort by CPU & Memory
        pods_metrics = [
            line.split() for line in result.stdout.strip().split("\n")
        ]
        top_cpu_pods = sorted(pods_metrics, key=lambda x: int(x[1].replace("m", "")), reverse=True)[:2]
        top_mem_pods = sorted(pods_metrics, key=lambda x: int(x[2].replace("Mi", "")), reverse=True)[:2]

        # Extract pod names and values
        cpu_pod1, cpu_val1 = top_cpu_pods[0][0], int(top_cpu_pods[0][1].replace("m", "")) // 10
        cpu_pod2, cpu_val2 = top_cpu_pods[1][0], int(top_cpu_pods[1][1].replace("m", "")) // 10

        mem_pod1, mem_val1 = top_mem_pods[0][0], int(top_mem_pods[0][2].replace("Mi", "")) // 10
        mem_pod2, mem_val2 = top_mem_pods[1][0], int(top_mem_pods[1][2].replace("Mi", "")) // 10

        # Clear the terminal
        subprocess.run("clear", shell=True)

        print(f"Top Pods by Resource Usage (Last Refresh: {time.strftime('%H:%M:%S')})")

        # Plot CPU and Memory charts
        plot("CPU (milliCPU)", cpu_pod1, cpu_pod2, [cpu_val1], [cpu_val2])
        print()
        plot("Memory (MiB)", mem_pod1, mem_pod2, [mem_val1], [mem_val2])

    except KeyboardInterrupt:
        print("\nStopping monitoring.")
        break

    time.sleep(5)


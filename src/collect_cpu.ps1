<#
.SYNOPSIS
  Collects CPU utilization for a specified duration at a specified sample interval,
  and writes results to CSV. CPU usage is normalized between [0..1].

.DESCRIPTION
  This script uses Get-Counter to read the "\Processor(_Total)\% Processor Time" counter,
  repeating until the specified duration has elapsed. The CPU utilization is then
  divided by 100 to yield a [0..1] range. Results (timestamp + CPU usage) are exported to CSV.

.PARAMETER Duration
  The total number of seconds to collect CPU usage data.

.PARAMETER SampleInterval
  The delay (in seconds) between each CPU usage sample.

.PARAMETER OutputCsv
  The file path where the CPU usage data will be written in CSV format.

.EXAMPLE
  CollectCpu.ps1 -Duration 60 -SampleInterval 2 -OutputCsv "cpu_utilization.csv"
#>

param(
    [Parameter(Mandatory=$true)]
    [int]$Duration,

    [Parameter(Mandatory=$true)]
    [int]$SampleInterval,

    [Parameter(Mandatory=$true)]
    [string]$OutputCsv
)

Write-Host "Collecting CPU usage for $Duration seconds (one sample every $SampleInterval s)..."
Write-Host "Results will be written to: $OutputCsv"
Write-Host ""

# We will store all results in a PowerShell array
$results = @()

# Calculate when we should stop sampling
$endTime = (Get-Date).AddSeconds($Duration)

while ((Get-Date) -lt $endTime) {

    # Record the time of this sample
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    # Get the CPU usage (CookedValue is the % usage [0..100])
    $counterData = Get-Counter -Counter "\Processor(_Total)\% Processor Time" -SampleInterval $SampleInterval
    $cpuPercent = $counterData.CounterSamples.CookedValue

    # Convert to [0..1] by dividing by 100
    $cpuUsage = $cpuPercent / 100.0

    # Create a small object with the fields we want
    $results += [PSCustomObject]@{
        Timestamp   = $timestamp
        CPU_Usage   = [math]::Round($cpuUsage, 4) # e.g. 0.4321
    }
}

# Export the array of objects to CSV
$results | Export-Csv -Path $OutputCsv -NoTypeInformation

Write-Host ""
Write-Host "Done! CPU usage data (0..1) has been written to $OutputCsv."


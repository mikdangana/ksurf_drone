<#
.SYNOPSIS
  Launches a program, prints its output to the console, and collects its memory usage into a CSV.
  The script only terminates after the launched program terminates.

.DESCRIPTION
  This script:
   1) Initializes a .NET Process object for the specified executable (ProgramPath + Arguments).
   2) Redirects stdout so we can capture and display it continuously.
   3) In a loop:
       - Reads any available lines from stdout
       - Collects memory usage of the process
       - Sleeps for SampleInterval seconds
     Exits the loop once the process has exited.
   4) Exports the memory usage data to CSV.

.PARAMETER ProgramPath
  The full path of the program (.exe) to launch.

.PARAMETER Arguments
  Optional arguments to pass to the program.

.PARAMETER SampleInterval
  Number of seconds between memory usage samples.

.PARAMETER OutputCsv
  The file path where CSV results will be written.

.EXAMPLE
  .\CollectMemoryUsageWithStdout.ps1 -ProgramPath "C:\Path\MyApp.exe" `
     -Arguments "--option foo" -SampleInterval 2 -OutputCsv "memory.csv"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ProgramPath,

    [Parameter(Mandatory=$false)]
    [string]$Arguments = "",

    [Parameter(Mandatory=$true)]
    [int]$SampleInterval,

    [Parameter(Mandatory=$true)]
    [string]$OutputCsv
)

Write-Host "Launching program: $ProgramPath $Arguments"
Write-Host "SampleInterval: $SampleInterval (s)"
Write-Host "Output CSV: $OutputCsv"

# -- Create & configure the process object
$proc = New-Object System.Diagnostics.Process
$proc.StartInfo.FileName               = $ProgramPath
$proc.StartInfo.Arguments              = $Arguments
$proc.StartInfo.UseShellExecute        = $false
$proc.StartInfo.RedirectStandardOutput = $true
$proc.StartInfo.CreateNoWindow         = $true

# Try to start the process
try {
    $started = $proc.Start()
    if (-not $started) {
        throw "Failed to start process $ProgramPath."
    }
} catch {
    Write-Error "Error launching process: $_"
    return
}

Write-Host "Process started (PID = $($proc.Id)). Collecting memory usage..."

# We'll store results in an array
$results = @()

# Define a helper function to read & print any currently available stdout lines
function Read-Stdout($processRef) {
    while (-not $processRef.StandardOutput.EndOfStream) {
        $line = $processRef.StandardOutput.ReadLine()
        Write-Host $line
    }
}

# Main loop:
# Continue while the process is alive. We'll poll memory usage every $SampleInterval seconds.
while (-not $proc.HasExited) {

    # 1) Read any available lines from stdout
    Read-Stdout $proc

    # 2) Check memory usage
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    # (We can still retrieve info from the running process via Get-Process or $proc object)
    try {
        # If the process is still alive, get memory usage
        $memInfo = Get-Process -Id $proc.Id -ErrorAction Stop
        $memMB   = [math]::Round($memInfo.WorkingSet64 / 1MB, 2)
    } catch {
        # If we can't get info (perhaps process exited?), break
        break
    }

    # 3) Append data to results array
    $results += [PSCustomObject]@{
        Timestamp = $timestamp
        PID       = $proc.Id
        Memory_MB = $memMB
    }

    # 4) Wait the sample interval
    Start-Sleep -Seconds $SampleInterval
}

# One last read for any remaining lines if the process wrote something before exit
Read-Stdout $proc

Write-Host "Process has exited."

# Export the array of objects to CSV
$results | Export-Csv -Path $OutputCsv -NoTypeInformation
Write-Host "`nDone! Memory usage data written to '$OutputCsv'."


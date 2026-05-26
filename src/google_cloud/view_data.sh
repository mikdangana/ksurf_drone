#!/bin/sh


#ROOT=/c/Users/mikda/Downloads/uoft/kalman_filter/google_cloud
ROOT=~/src/kfpca/src/google_cloud

OUT=$ROOT/out.txt

OUTCSV=$ROOT/outcsv.txt

LINES=1000000


parse_trace_file() {
    #CMD="/c/Users/mikda/Downloads/DynamoRIO-Windows-10.0.0/DynamoRIO-Windows-10.0.0/bin64/drrun -t drcachesim -simulator_type view -indir $ROOT/data -sim_refs $LINES"
    CMD="$ROOT/DynamoRIO-Linux-10.0.0/bin64/drrun -t drcachesim -simulator_type view -indir $ROOT/data -sim_refs $LINES"

    RES=`$CMD 1>$OUT 2>>$OUT`

    RES=`tail -$((LINES+2)) $OUT | head -$LINES > $OUTCSV`

    echo executed CMD: $CMD; echo
    echo "output: $OUT & $OUTCSV"
}


parse_trace_file


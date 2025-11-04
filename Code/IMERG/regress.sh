#!/bin/sh

PYTHON_SCRIPT="regress.py"

for i in {0..5}; do

    init=$((2*i+1))
    end=$((2*i+3))
    echo $init $end "start"
    python $PYTHON_SCRIPT $init $end
    echo $init $end "done"
done
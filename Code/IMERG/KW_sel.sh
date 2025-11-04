#!/bin/sh

PYTHON_SCRIPT="KW_sel.py"

for i in {0..5}; do

    init=$((2*i+1))
    end=$((2*i+3))
    echo $init $end "start"
    python $PYTHON_SCRIPT $init $end
done
#!/bin/sh

PYTHON_SCRIPT="corr_composite.py"

# for KW
for i in {0..5}; do
    init=$((2*i+1))
    end=$((2*i+3))

    echo $init $end "start"
    python $PYTHON_SCRIPT $init $end "kw"
    echo $init $end "done"
done

# for mjo
echo "mjo state"
python $PYTHON_SCRIPT 1 4 "mjo"
echo "mjo end"

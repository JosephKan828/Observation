#!/bin/sh

PYTHON_SCRIPT="CloudSat_regress.py"

# for KW
# for i in {0..5}; do
#     for l in {0..18}; do
#         init=$((2*i+1))
#         end=$((2*i+3))
#         lon=$((20*l))
#         echo $init $end $lon "start"
#         python $PYTHON_SCRIPT $init $end $lon
#         echo $init $end $lon "done"
#     done
# done

# for MJO
for l in {0..18}; do
    lon=$((20*l))
    echo $lon "start"
    python $PYTHON_SCRIPT 1 4 $lon
    echo $lon "done"
done
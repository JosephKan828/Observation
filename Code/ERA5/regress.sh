#!/bin/sh

PYTHON_SCRIPT=/home/b11209013/2025_Research/Obs/Code/ERA5/regress.py

FILES=(/home/b11209013/2025_Research/Obs/Files/IMERG/prec*.h5)

for file in ${FILES[@]}; do

	for l in {0..18}; do
		lon=$((20*l))

		echo $file $lon start

		python $PYTHON_SCRIPT $file $lon

		echo $file $lon end
	done

done

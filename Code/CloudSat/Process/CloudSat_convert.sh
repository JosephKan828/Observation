#!/bin/bash

for year in {2006..2017}; do
	for day in $(seq 1 366); do
		python CloudSat_convert.py "$year" "$day"
	done
done

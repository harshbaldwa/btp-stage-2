#!/bin/bash
for p in 6 12 24
do
    echo $p
    for i in 10000 20000 40000 100000 200000 500000 750000 1000000 2000000 5000000
    do
        echo $i
	python main-electric.py -n $i -p $p -b opencl
    done
done

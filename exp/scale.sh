#!/bin/bash
# training-size check: hold out series 0, drop one more training series each time
for d in 1 2 3 4; do
  python exp/e01.py --model xc --res 32 --unary 1 --epochs 40 --vshift 8 --folds 0 --also_drop $d --test 0 --name scale_$d | grep -E "ep 40|scores"
done

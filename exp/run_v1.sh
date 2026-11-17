#!/bin/bash
# fresh run of submissions/v1 with the platform's command line
set -e
rm -rf working && mkdir working
python submissions/v1/solution.py dataset/public working/submission.csv
cp working/submission.csv $ERIS_OUT/

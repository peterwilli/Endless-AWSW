#!/bin/bash
for f in *.ipynb
do
  echo "Processing $f..."
  jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 --inplace "$f"
done
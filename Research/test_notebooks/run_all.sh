#!/bin/bash
for f in *.ipynb
do
  echo "Processing $f..."
  runipy -o "$f"
done
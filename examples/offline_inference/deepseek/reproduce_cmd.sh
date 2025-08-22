#!/bin/bash
echo "Starting test with const scale_format"
bash run_ds.sh

sleep 10
echo "Starting test with scalar scale_format and patching enabled"
bash run_ds.sh --use-scalar --patching 

sleep 10
echo "Starting test with scalar scale_format without patching"
bash run_ds.sh --use-scalar

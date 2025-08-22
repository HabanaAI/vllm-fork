#!/bin/bash

bash run_ds.sh

sleep 10
bash run_ds.sh --use-scalar --patching 

sleep 10
bash run_ds.sh --use-scalar

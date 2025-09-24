#!/bin/bash

# curl -X POST http://localhost:12345/start_profile
python client_updated.py --prompt member -f 50-pages/50-pages/ -o result
python client_updated.py --prompt validate -f 50-pages/50-pages/ -o result
# curl -X POST http://localhost:12345/stop_profile
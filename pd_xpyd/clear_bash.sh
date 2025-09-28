#!/bin/bash
ps -eo pid,comm | awk -v self=$$ -v parent=$PPID '$2=="bash" && $1!=1 && $1!=self && $1!=parent {print $1}' | xargs -r kill -9


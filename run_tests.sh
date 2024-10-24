#!/usr/bin/bash
torchrun  --nproc_per_node=1 -m pytest  tests/unit_tests/test_basic.py

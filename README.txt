Code for CSE 534 Analysis of Privacy Leakage in Secure DNS using Neural Network Classifiers

Running different scripts:
The scripts will tell you how to run them if run with no arguments. Any script not listed here is not intended to be run by hand.
Summary:
- ./trace_file.sh <csv_file_name> <port_number>
-- This file captures pcap traces. It's set up to run on my specific machine, though the machine-specific parts can be changed easily enough.
- python3 trace_analysis.py <category> <doq|doh>
-- This script analyzes each pcapng trace in data/traces/<category>/<doq|doh> and saves its output in output.csv in the same directory. 
- python3 trees.py <filename> <tree|extra|forest|adaboost> [<transfer-learning filename>]
-- This file runs the tree-based networks on a dataset CSV file. Optionally provide a different dataset to validate transfer learning.
- python3 main.py [-train {csv_path}] [-load {model_path} {test}]
-- This file handles training and validation of deep learning. Most of the other deep learning related files (model.py, runner.py, pipeline.py) are executed by this script.


Feature datasets are stored in data/csv/
Traces are stored in data/traces/ (broken down by category into doq/doh traces in subdirectories)
Lists of sites are stored in data/sites/
Deep Learning models are stored in models/
Raw results are stored in data/results/
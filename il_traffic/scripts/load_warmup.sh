#!/bin/bash

# Default values of arguments
SCRIPT_DIRECTORY=$PWD/$(dirname "$0")
IL_TRAFFIC_DIRECTORY=$SCRIPT_DIRECTORY/../..

# Loop through arguments and process them.
for arg in "$@"
do
  case $arg in
    -h|--help)
    echo "Loads warmup files used to run simulations from a predefined initial state."
    echo ""
    echo "WARNING: Existing files within the warmup/ folder will be removed by this script."
    echo ""
    echo "usage: load_warmup.sh [--help]"
    echo ""
    echo "arguments"
    echo "  -h, --help            show this help message and exit"
    exit 0
    ;;
    *)
    OTHER_ARGUMENTS+=("$1")
    shift # Remove generic argument from processing
    ;;
  esac
done

# Create the warmup folder.
if [ ! -d $IL_TRAFFIC_DIRECTORY/"warmup" ]; then
  mkdir $IL_TRAFFIC_DIRECTORY/"warmup"
fi
cd $IL_TRAFFIC_DIRECTORY/"warmup"

# Download and extract highway warmup files.
wget https://berkeley.box.com/shared/static/t7pbo49rxplor1fv1jczgv9cczu4bwg2.gz
tar -xvzf t7pbo49rxplor1fv1jczgv9cczu4bwg2.gz && rm t7pbo49rxplor1fv1jczgv9cczu4bwg2.gz

# Download and extract I-210 warmup files.
wget https://berkeley.box.com/shared/static/99o6sboo6p19che1q0gpbbzgk93avvw7.gz
tar -xvzf 99o6sboo6p19che1q0gpbbzgk93avvw7.gz && rm 99o6sboo6p19che1q0gpbbzgk93avvw7.gz

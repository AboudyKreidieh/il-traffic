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
# TODO: download
tar -xvzf highway.tar.gz && rm highway.tar.gz

# Download and extract I-210 warmup files.
# TODO: download
tar -xvzf i210.tar.gz && rm i210.tar.gz

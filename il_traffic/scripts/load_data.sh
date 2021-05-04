#!/bin/bash

# default values of arguments
SCRIPT_DIRECTORY=$PWD/$(dirname "$0")
IL_TRAFFIC_DIRECTORY=$SCRIPT_DIRECTORY/../..
MODEL="all"
FULL="False"

# Loop through arguments and process them.
for arg in "$@"
do
  case $arg in
    -h|--help)
    echo "Loads preexisting simulations from baseline, expert, imitated models."
    echo ""
    echo "This script also loads the learned checkpoint from the derived models."
    echo "See the README within the document for more."
    echo ""
    echo "WARNING: Existing files within the data/ folder will be removed by this script."
    echo ""
    echo "usage: load_data.sh [--model MODEL] [--full] [--help]"
    echo ""
    echo "arguments"
    echo "  -h, --help            show this help message and exit"
    echo "  --model MODEL         the model to load. One of {baseline, expert, imitated}. If not"
    echo "                        specified, all models are downloaded."
    echo "  --full                apply list or eval only to the hard examples"
    exit 0
    ;;
    --full)
    FULL="True"
    shift
    ;;
    --model)
    MODEL="$2"
    shift # Remove argument name from processing
    ;;
    *)
    OTHER_ARGUMENTS+=("$1")
    shift # Remove generic argument from processing
    ;;
  esac
done

# Run assertions on choice of model.
if [ $MODEL != "all" ] && [ $MODEL != "baseline" ] && [ $MODEL != "expert" ] && [ $MODEL != "imitated" ]; then
  echo "Error: Unknown model: $MODEL. Quitting."
  exit 1
fi

# Create the data folder.
if [ ! -d $IL_TRAFFIC_DIRECTORY/"data" ]; then
  mkdir $IL_TRAFFIC_DIRECTORY/"data"
fi
cd $IL_TRAFFIC_DIRECTORY/"data"

# Download and unzip the requested data.
if [ $MODEL == "all" ] && [ $MODEL == "baseline" ]; then
  echo "Downloading baseline data."
  if [ $FULL == "True" ]; then
    continue  # TODO: download
  else
    continue  # TODO: download
  fi
  echo "Unzipping baseline data."
  tar -xvzf baseline.tar.gz && rm baseline.tar.gz
if [ $MODEL == "all" ] && [ $MODEL == "expert" ]; then
  echo "Downloading expert data."
  if [ $FULL == "True" ]; then
    continue  # TODO: download
  else
    continue  # TODO: download
  fi
  echo "Unzipping expert data."
  tar -xvzf expert.tar.gz && rm expert.tar.gz
if [ $MODEL == "all" ] && [ $MODEL == "imitated" ]; then
  echo "Downloading imitated data."
  if [ $FULL == "True" ]; then
    continue  # TODO: download
  else
    continue  # TODO: download
  fi
  echo "Unzipping imitated data."
  tar -xvzf imitated.tar.gz && rm imitated.tar.gz
fi

exit 0

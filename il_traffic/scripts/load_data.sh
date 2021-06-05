#!/bin/bash

# default values of arguments
SCRIPT_DIRECTORY=$PWD/$(dirname "$0")
IL_TRAFFIC_DIRECTORY=$SCRIPT_DIRECTORY/../..
MODEL="all"
PENETRATION_RATE="0.05"
FULL="False"
TRANSFER="False"

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
    echo "usage: load_data.sh [--model MODEL] [--penetration PEN] [--full] [--transfer] [--help]"
    echo ""
    echo "arguments"
    echo "  -h, --help            show this help message and exit"
    echo "  --model MODEL         the model to load. One of {baseline, expert, imitated-1, "
    echo "                        imitated-5}. If not specified, all models are downloaded."
    echo "  --penetration PEN     the penetration rate. One of {0.025, 0.05, 0.075, 0.1}. Ignored"
    echo "                        if --model is set to \"baseline\", or --transfer is set."
    echo "  --full                whether to download the full or partial trajectories. The full"
    echo "                        trajectories also include the emission.csv file."
    echo "  --transfer            whether to download the transfer tests. Only penetration_rate"
    echo "                        of 0.05 is valid in this case"
    exit 0
    ;;
    --full)
    FULL="True"
    shift
    ;;
    --transfer)
    TRANSFER="True"
    shift
    ;;
    --model)
    MODEL="$2"
    shift # Remove argument name from processing
    ;;
    --penetration_rate)
    PENETRATION_RATE="$2"
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

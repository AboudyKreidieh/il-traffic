# Learning Energy-Efficient Driving Behaviors by Imitating Experts

Implementation of the code for the paper titled "Learning Energy-Efficient 
Driving Behaviors by Imitating Experts". To learn more, see the following 
links:

- **Paper:** TODO
- **Website:** https://sites.google.com/view/il-traffic/home

**TODO: GIF without control ---- GIF with control**

## Contents

1. [Setup Instructions](#1-setup-instructions)  
    1.1. [Basic Installation](#11-basic-installation)  
    1.2. [Docker Installation](#12-docker-installation)  
    1.3. [Downloading Warmup States](#13-downloading-warmup-states)  
2. [Usage](#2-usage)  
    2.1. [Simulating Baseline and Expert Models](#21-simulating-baseline-and-expert-models)  
    2.2. [Imitating Experts](#22-imitating-experts)  
    2.3. [Evaluating Results](#23-evaluating-results)  
3. [Visualizing Existing Data](#3-visualizing-existing-data)  
    3.1. [Downloading Models and Results](#31-downloading-models-and-results)  
    3.2. [Visualizing Behaviors and Trajectories](#32-visualizing-behaviors-and-trajectories)  
4. [Citing](#4-citing)

## 1. Setup Instructions

### 1.1 Basic Installation

This repository is an extension of the [Flow](TODO) repository. If you have not 
previously installed Flow, begin by following the setup instruction provided 
[here](TODO).

Once Flow has been installed, open a terminal and set the working directory of
the terminal to match the path to this repository:

```shell script
cd path/to/il-traffic
```

If you have installed Flow in conda environment, you will want to install this
repository in the same environment. If you followed the basic Flow setup 
instructions, this can be done my running the following command:

```shell script
source activate flow
```

Finally, install the contents of the repository onto your conda environment (or
your local python build) by running the following command:

```shell script
pip install -e .
```

If you would like to (optionally) validate that the repository successfully
installed and is running, you can do so by executing the unit tests as follows:

```shell script
nose2
```

The test should return a message along the lines of:

    ----------------------------------------------------------------------
    Ran XXX tests in YYYs

    OK

### 1.2 Docker Installation

TODO

### 1.3 Downloading Warmup States

Warmup states provide initializations to the positions and speeds of vehicles 
within a given network. These states allow us to subvert the need to run 
multiple "warmup" simulation steps to allow for the onset of congestion to 
occur. For this repository, we've created warmup files for both the "highway" 
and "i210" networks. These files are:

1. taken after 3600 seconds worth of simulation steps 
2. taken for inflow rates ranging from 1900 to 2300 veh/hr/lane in increments 
   of 50
3. taken for downstream speed limits ranging from 5 to 7 m/s in increments of 1

To install the warmup file programmatically, run from the base directory:

```shell script
il_traffic/scripts/load_warmup.sh
```

This will create a new folder in the base directory called "warmup" with to 
additional sub-folders called "highway" and "i210" which contain the warmup 
files, and a description.csv file. If this operation is successful, all 
[simulations](#21-simulating-baseline-and-expert-models) and 
[evaluations](#23-evaluating-results) can now be run using the `--use_warmup` 
flag.

**Note:** If you would rather download the files separately, you can click on
the individual links below:

| Network   | Links          |
|-----------|----------------|
| highway   | [click here]() |
| i210      | [click here]() |

## 2. Usage

TODO

### 2.1 Simulating Baseline and Expert Models

TODO

### 2.2 Imitating Experts

TODO

### 2.3 Evaluating Results

TODO

## 3. Visualizing Existing Data  

TODO

### 3.1 Downloading Models and Results

The trained models and trajectories provided within the paper and website are 
available to be downloaded and further analyzed. To download the existing 
models and trajectories, run:

```shell script
il_traffic/scripts/load_data.sh
```

where the descriptions to additional parameters can be read by running:

```shell script
il_traffic/scripts/load_data.sh --help
```

The script will create a "data" folder with all the relevant models and/or 
trajectories downloaded. The individual folders will contain content similar to
what is produced by the `simulate.py` and `evaluate.py` scripts.

**Note:** If you would like to install the trajectories and models via 
separate links, you can do so from the below tables:

* **Trained models:**

| Controller          | Model (5 seeds) |
|---------------------|-----------------|
| Imitated (1 frame)  | [1]() - [2]() - [3]() - [4]() - [5]() |
| Imitated (5 frames) | [1]() - [2]() - [3]() - [4]() - [5]() |

-----------------

* **Trajectories for different penetration rates:**

| Controller          | Penetration Rate | Trajectories (5 seeds) |
|---------------------|------------------|------------------------|
| Baseline            | 0 %              | [1]() - [2]() - [3]() - [4]() - [5]() |
| Follower Stopper    | 2.5 %            | [1]() - [2]() - [3]() - [4]() - [5]() |
|                     | 5.0 %            | [1]() - [2]() - [3]() - [4]() - [5]() |
|                     | 7.5 %            | [1]() - [2]() - [3]() - [4]() - [5]() |
|                     | 10.0 %           | [1]() - [2]() - [3]() - [4]() - [5]() |
| Imitated (1 frame)  | 2.5 %            | [1]() - [2]() - [3]() - [4]() - [5]() |
|                     | 5.0 %            | [1]() - [2]() - [3]() - [4]() - [5]() |
|                     | 7.5 %            | [1]() - [2]() - [3]() - [4]() - [5]() |
|                     | 10.0 %           | [1]() - [2]() - [3]() - [4]() - [5]() |
| Imitated (5 frames) | 2.5 %            | [1]() - [2]() - [3]() - [4]() - [5]() |
|                     | 5.0 %            | [1]() - [2]() - [3]() - [4]() - [5]() |
|                     | 7.5 %            | [1]() - [2]() - [3]() - [4]() - [5]() |
|                     | 10.0 %           | [1]() - [2]() - [3]() - [4]() - [5]() |

-----------------

* **Trajectories from robustness tests:**

| Controller          | Trajectories (5 seeds) |
|---------------------|------------------------|
| Baseline            | [1]() - [2]() - [3]() - [4]() - [5]() |
| Follower Stopper    | [1]() - [2]() - [3]() - [4]() - [5]() |
| Imitated (1 frame)  | [1]() - [2]() - [3]() - [4]() - [5]() |
| Imitated (5 frames) | [1]() - [2]() - [3]() - [4]() - [5]() |

### 3.2 Visualizing Behaviors and Trajectories

TODO

## 4. Citing

TODO

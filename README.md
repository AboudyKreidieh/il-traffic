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

TODO

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

TODO

### 3.2 Visualizing Behaviors and Trajectories

TODO

## 4. Citing

TODO

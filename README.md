# Learning Energy-Efficient Driving Behaviors by Imitating Experts

Implementation of the code for the paper titled "Learning Energy-Efficient 
Driving Behaviors by Imitating Experts". To learn more, see the following 
links:

- **Paper:** TODO
- **Website:** https://sites.google.com/view/il-traffic/home

<p align="center"><img src="docs/img/uncontrolled.gif" width="48%"/> &emsp; <img src="docs/img/controlled.gif" width="48%"/></p>
<p align="center">Imitation results on I-210. <b>Left:</b> baseline with stop-and-go waves. <b>Right:</b> imitated policy allowing for wave dissipation </p>

## Contents

1. [Setup Instructions](#1-setup-instructions)  
    1.1. [Basic Installation](#11-basic-installation)  
    1.2. [Docker Installation](#12-docker-installation)  
    1.3. [Downloading Warmup States](#13-downloading-warmup-states)  
2. [Usage](#2-usage)  
    2.1. [Simulating Baseline and Expert Models](#21-simulating-baseline-and-expert-models)  
    2.2. [Imitating Experts](#22-imitating-experts)  
    2.3. [Evaluating Results](#23-evaluating-results)  
    2.4. [Downloading Models and Results](#24-downloading-models-and-results)  
3. [Citing](#3-citing)

## 1. Setup Instructions

### 1.1 Basic Installation

This repository is an extension of the [Flow](https://flow-project.github.io/)
repository. If you have not previously installed Flow, begin by following the 
setup instruction provided 
[here](https://flow.readthedocs.io/en/latest/flow_setup.html).

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
| highway   | [click here](https://berkeley.box.com/shared/static/t7pbo49rxplor1fv1jczgv9cczu4bwg2.gz) |
| i210      | [click here](https://berkeley.box.com/shared/static/99o6sboo6p19che1q0gpbbzgk93avvw7.gz) |

## 2. Usage

We describe in the following subsections how different hand-designed baseline 
and AV (expert) models can be simulated within different networks, and describe
the imitation and evaluation procedures. Results from previous runs using this
repository can further be downloaded and visualized through the final 
subsection.

### 2.1 Simulating Baseline and Expert Models

Through this repository, simulations of both baseline (human-driven) behaviors 
and mixed-autonomy behaviors in which AVs follow a variety of different 
controllers can be conducted through the `simulate.py` script. The networks 
explored in this repository, see the figure below, include a single lane 
highway and simulated version of the I-210 network. A description of the 
process through which congestion forms in these model is available in our 
[paper](TODO).

<p align="center"><img src="docs/img/networks.png" align="middle" width="100%"/></p>

To execute a simulation of the network, run:

```shell
python il_traffic/scripts/simulate.py
```

where the descriptions to additional arguments can be seen by running:

```shell script
python il_traffic/scripts/simulate.py --help
```

> **Note:**  If you are using the `--use_warmup` flag, be sure to download the 
> warmup files first, see [this section](#13-downloading-warmup-states).

The above script will start a simulation of the network that can be visualized 
if `--render` is set. Moreover, if `--gen_emission` is set, this script will 
create a folder in "expert_data/{network}/{controller}/{inflow}-{end_speed}" 
containing the following files:

* avg-speed.png : a plot of the avg/std speeds of all vehicles at every time 
  step.
* emission.csv : the trajectory data collected from the simulation, containing 
  values that denote the speed, position, and accelerations conducted by all 
  vehicles at all time steps.
* mpg.csv : the energy values each individual vehicle experiences after moving 
  forward for 50 meters (in miles-per-gallon, or mpg).
* mpg.png : a plot of the mpg values contained in mpg.csv, with a line plot 
  used to represent the average values across time.
* ts-{0-4}.png : visualization of the trajectories of individual vehicles as 
  seen as a time-space diagram on each individual lane. The number after the 
  dash represents the lane number (0 for the highway and 0-4 for the I-210).
* tt.json : the time it takes every vehicle to traverse the network to the 
  downstream edge.

### 2.2 Imitating Experts

The behaviors of the baseline and expert controllers presented in the 
subsection above can be imitated to a neural network policy (or an ensemble of
policies) through the `imitate.py` method in the "scripts" folder. This 
method implements the DAgger algorithm, and provides additional augmentations 
to allow for the training of ensembles of (optionally stochastic) policies, as 
well as various other features such as dropout and batch normalization. To 
start the imitation procedure, run:

```shell script
python il_traffic/scripts/imitate.py
```

where the descriptions to additional arguments can be seen by running:

```shell script
python il_traffic/scripts/imitate.py --help
```

Once the imitation procedure has begun, it will create an "imitation_data" 
folder which will store the trained model after every training iteration. The 
folder will also contain a tensorboard log and "train.csv" file that describe 
the performance of the model at every iteration.

### 2.3 Evaluating Results

Once a given expert has been imitated, the performance of the model can be 
verified through the `evaluate.py` method by running:

```shell script
python il_traffic/scripts/evaluate.py "/path/to/results_folder"
```

where the first argument is the path to the folder created by the imitation 
method before, and the additional arguments can be seen by running:

```shell script
python il_traffic/scripts/evaluate.py --help
```

> **Note:**  If you are using the `--use_warmup` flag, be sure to download the 
> warmup files first, see [this section](#13-downloading-warmup-states).

If the `--gen_emission` flag has been set, the script will create a new 
"results" folder in the original folder with the model containing trajectory 
data similar to the one created by the
[simulation procedure](#21-simulating-baseline-and-expert-models).

### 2.4 Downloading Models and Results

The trained models and trajectories provided within the paper and website are 
available to be downloaded and further analyzed. To download the existing 
models and trajectories, run:

```shell script
il_traffic/scripts/load_data.sh
```

where the descriptions to additional arguments can be seen by running:

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
| Imitated (1 frame)  | [1](https://berkeley.box.com/shared/static/ueyl2857e199rqee3k9mr7zsfg1lztky.gz) - [2](https://berkeley.box.com/shared/static/8t24lxu8igpmk1jv8nakojy7hrg72y12.gz) - [3](https://berkeley.box.com/shared/static/su1s2unsotcs0xy08c2x1xug3sjdesuw.gz) - [4](TODO) - [5](TODO) |
| Imitated (5 frames) | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |

* **Trajectories for different penetration rates:**

| Controller          | Penetration Rate | Trajectories (5 seeds) |
|---------------------|------------------|------------------------|
| Baseline            | 0 %              | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
| Follower Stopper    | 2.5 %            | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
|                     | 5.0 %            | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
|                     | 7.5 %            | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
|                     | 10.0 %           | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
| Imitated (1 frame)  | 2.5 %            | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
|                     | 5.0 %            | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
|                     | 7.5 %            | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
|                     | 10.0 %           | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
| Imitated (5 frames) | 2.5 %            | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
|                     | 5.0 %            | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
|                     | 7.5 %            | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
|                     | 10.0 %           | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |

* **Trajectories from robustness tests:**

| Controller          | Trajectories (5 seeds) |
|---------------------|------------------------|
| Baseline            | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
| Follower Stopper    | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
| Imitated (1 frame)  | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |
| Imitated (5 frames) | [1](TODO) - [2](TODO) - [3](TODO) - [4](TODO) - [5](TODO) |

## 3. Citing

To cite this repository in publications, use the following:

TODO

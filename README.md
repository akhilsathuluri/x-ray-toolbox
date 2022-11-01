X-Ray toolbox
========

[![Github license](https://img.shields.io/github/license/akhilsathuluri/x-ray-tool)](https://github.com/akhilsathuluri/x-ray-tool)
[![python](https://img.shields.io/badge/python-3.8-green)](https://github.com/akhilsathuluri/x-ray-tool)
[![Github release](https://img.shields.io/github/release/akhilsathuluri/x-ray-tool)](https://github.com/akhilsathuluri/x-ray-tool/releases)
[![Github issues](https://img.shields.io/github/issues/akhilsathuluri/x-ray-tool)](https://github.com/akhilsathuluri/x-ray-tool)

![Top down design process](https://github.com/akhilsathuluri/x-ray-tool/blob/main/screenshots/top-down-design.png)

X-Ray toolbox is written in Python and allows users to break down system level requirements of a multi-component system to individual component level requirements. The library is developed and maintained by the [Robot Systms Group, Laboratory for Product Development and Lightweight Design, TU Munich](https://www.mec.ed.tum.de/en/lpl/research/research-groups/robot-systems/). It allows users to understand and interpret the trade-offs while designing a system with several interdependant sub-systems. 



To dive more into the theory have a look at the paper, [Computing solution spaces for robust design](https://github.com/PhD-TUM/xray-python/files/7787066/file.pdf) and [On the design of large systems subject to uncertainty](https://www.tandfonline.com/doi/abs/10.1080/09544828.2017.1303664). 


## Requirements
The library is tested on Ubuntu 20.04 with Python 3.8

## Installation
* Download the released version or clone the repository 
* Dependancies can be installed using the `requirements.txt` file

## Examples
Two examples are provided in [`src\problems`](https://github.com/akhilsathuluri/x-ray-tool/tree/main/src/problems) to demonstrate the working of the tool and problem setup. They are:
* [`CrashDesign`](https://github.com/akhilsathuluri/x-ray-tool/tree/main/src/problems/CrashDesign): shows decomposing requirements for a car crash problem
* [`Line`](https://github.com/akhilsathuluri/x-ray-tool/tree/main/src/problems/Line): shows setting up the problem for a dummy system 

## Usage
* To setup a new problem go to: `xray-main/src/problems`
* All problems follow similar template as the `Line` problem given as an example, so you can copy paste its directory to get started
* Name your problem as the name of the folder (DO NOT USE SPACES IN THE NAME)
* Go into the folder and setup your design variable and quantities of interest bounds in the input folder in their respective .csv files and do not use mathematical expressions, please define numbers
* Use simple variable names, preferably a single letter, also give a description for the variable for clarity
* Similarly, setup bounds on your quantities of interest
* All quantities of interest and design variables should have an upper and lower bound (if you have one variable or discrete set of bounds please see issues to track progress on that feature)
* All bottom-up mappings are defined in the `library` directory
* Change the main file (\`Line.py\`) file to the same name as your directory
* Also change the class name to your directory name
* You can use your favorite editor to setup your bottom-up mappings (Like Atom: <https://atom.io/)>
* You can define problem description and problem name
* Also setup the plotter array to plot variables required (starting from first variable as 0), for example if you want plots for variables 1 and 2 add \[0, 1\] in the array plotter.

## Setup

* Once you setup your problem, load it by typing `streamlit run main.py` in the Anaconda prompt being in the `xray/src` directory
* You can find help by clicking the `+ Getting started` button in the center
* Click on your problem in the dropdown in `Problem Configuration`
* Define functions for each quantity of interest defined, make sure the name of the function is same as the name of the variable defined in qoi csv

## Help
* For general help look at the `Help` dropdown within the tool by clicking the `+` on the top of the screen
* You can also raise an issue directly in GitHub/GitLab or mail me at akhil.sathuluri@tum.de

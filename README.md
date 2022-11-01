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
* The main directory to setup new problems is `xray-main/src/problems`
* Name the folder same as the problem and define the domain for the design variables and the quantities of interest as done in [`CrashDesign`](https://github.com/akhilsathuluri/x-ray-tool/tree/main/src/problems/CrashDesign)
* The `bottom-up` mappings are defined in the `library` directory
* Define the desired design sections as an array in `plotter` within the problem definition

## Interface
* The interface is written in [streamlit](https://streamlit.io/) and can be accessed by,
```
streamlit run main.py
```
* You can find help by in the `+ Getting started` drop down
* Select problem from the `Problem Configuration` dropdown

## Help
For general use the `Help` dropdown within the tool by clicking the `+` on the top or raise an [issue](https://github.com/akhilsathuluri/x-ray-tool/issues) for help

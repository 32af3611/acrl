# README #

* Project: CFD with Reinforcement Learning
* Date: 19.01.2022

## General ##

These codes allow the preprocessing, processing and postprocessing of openfoam cases for 2D airfoil
simulations

## How-to Start ##

Install Openfoam7

### Setup Python Virtual Environment ###

Linux:

* Create a a Virtual Environment

  python3 -m venv .venv

* Active the Virtual Environment

  source .venv/bin/activate

* Install necessary packages

  pip install -r requirements

### bwunicluster ###

when submitting a job to the queue:

* Activate Python virtual Environment

* Load module cae/openfoam/7

* execute: foamInit

* continue with other calls, e.g. python scripts

Example of how to submit a job: "run_uc2"

### Run CFDblackbox from Python ###

#### Standalone ####

* generate and run a Test case:

  python3 cfdControl.py

#### In other python framework ####

* Call cfdBlackbox(caseID, caseDesc, blc_loc_SS, blc_coef_SS,blc_loc_PS, blc_coef_PS) in
  cfdControl.py



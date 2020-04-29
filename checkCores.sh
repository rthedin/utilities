#!/bin/bash

# ------------------------------------------------------------------- #
# Script to simply print out the lines that contain number of cores
# requested and check that agains `setUp` files. Visual check.
#
# Regis Thedin
# April 23, 2020
# regis.thedin@nrel.gov
# ------------------------------------------------------------------- #

checkCores(){

    grep -H --no-message "nCores" setUp*
    grep -H --no-message "ntasks" 1_*
    grep -H --no-message "ntasks" 2_*
    grep -H --no-message "ntasks" 3_*
    grep -H --no-message "ntasks" 4_*
    
    grep -H --no-message "cores=" 1_*
    grep -H --no-message "cores=" 2_*
    grep -H --no-message "cores=" 3_*
    grep -H --no-message "cores=" 4_*

}

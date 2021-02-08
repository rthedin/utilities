#!/bin/bash

# ------------------------------------------------------------------- #
# isolateTimeSteps.sh
#
# Script to copy timestep(s) from a decomposed case and save into
# a different directory. Useful for transfering and opening locally.
#
# Instructions: call from within the directory you want to isolate the
#          time-steps from. Provide a list of time steps to be isolated
# 
# Example: isolateTimeSteps 1000 1500 2000
#
# Regis Thedin
# Jul 14, 2020
# regis.thedin@nrel.gov
# ------------------------------------------------------------------- #

isolateTimeSteps(){
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color

    if [ "$#" -eq 0 ]; then
        echo -e "${RED}Error. ${NC}Supply the time-step(s) to be isolated (space-separated)."
        return
    fi

    orig=$(pwd)
    target="$orig"/../"$(basename $orig)"_SELECTdT
    cores=$(find $orig -maxdepth 1 -type d -name "processor*" | wc -l)

    if [ -d $target ]; then
        echo -e "${RED}$(basename $target) already exists. Stopping.${NC}\n"
        return
    fi

    echo "Preparing case for local analysis"
    echo -e "${GREEN}Source:${NC} $(basename $orig)"
    echo -e "${GREEN}Target:${NC} ../$(basename $target)"
    echo "Number of processor directories:" $cores

    mkdir $target
    cd $target

    echo "Copying 0, system, constant..."
    mkdir constant
    cp -rf $orig/{0.ori*,system} .
    cp -rf $orig/constant/{ABLPr*,cellD*,dynam*,g,polyMesh,transp*,turbu*} constant/

    # Check for non-valid times
    for arg; do
        if [ ! -d $orig/processor0/$arg ]; then
            echo -e "${RED}Time $arg does not exist. Skipping.${NC}"
        fi
    done

    # Main loop copying files
    for (( c=0; c<$cores; c++ )); do
        mkdir processor$c
        echo -ne "Copying processor$c/constant\r"
        cp -r $orig/processor$c/constant processor$c/
        for arg; do
            if [ -d $orig/processor0/$arg ]; then
                echo -ne "Copying processor$c/$arg        \r"
                cp -r $orig/processor$c/$arg/ processor$c
            fi
        done
    done

    cd $orig
    unset orig
    unset target

    echo -e "\n${GREEN}Done.${NC}"
}

#!/bin/bash

# ------------------------------------------------------------------- #
# Script to delete all intermediate time steps of an OpenFOAM case
# It laaves the last time directory and 0.
#
# Regis Thedin
# Oct 15, 2020
# regis.thedin@nrel.gov
# ------------------------------------------------------------------- #

rmintermediateDt(){

    latestTime=$(foamListTimes -processor -latestTime)
    if [ -d processor0 ]; then
        foamListTimes -rm -processor -time 1:$(($latestTime-1))
    fi

}

#!/bin/bash

# ----------------------------------------------------------------------- #
# copyCaseInstantZero.sh
#
# Copy all the files and settings from a different directory, but instead
# of simply copying everything, it only copies the first time step (that
# contains the mesh) and log files. No postProcessing or dynamicCode dirs.
#
# Regis Thedin
# Aug 25, 2020
# regis.thedin@nrel.gov
# ----------------------------------------------------------------------- #

copyCaseInstantZero(){

    GREEN='\033[0;32m'
    NC='\033[0m'

    echo  "Copying" $1 "into the current dir,"
    echo  "including processor*/0 and log files."
    
    cp -r $1/{1_*,2_*,3_*,4_*,setUp.*} . 2>/dev/null || :
    cp    $1/log.1.* .
    cp    $1/foam1* .
    cp -r $1/constant .
    cp -r $1/{system,0.original} .
    
    nCores=$(find $1 -maxdepth 1 -type d -name 'processor*' | wc -l)
    
    for (( c=0; c<nCores; c++ )); do
        mkdir processor$c
        cp -r $1/processor$c/constant processor$c/
        echo -ne "Copying processor$c       \r"
        cp -r $1/processor$c/0        processor$c/
    done

    echo -e "${GREEN} Done.               ${NC}"    
}


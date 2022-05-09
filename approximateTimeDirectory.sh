#!/bin/bash

# ----------------------------------------------------------------------- #
# approximateTimeDirectory.sh
#
# Approximate time directories to nearest decimal. Starting, ending and
# intervals should be given.
#
# Example call:
# approximateTimeDirectory 14400 14401.8 0.2
# where 14400.000000011 and 14401.800000012 exist.
#
# Regis Thedin
# May 8, 2022
# regis.thedin@nrel.gov
# ----------------------------------------------------------------------- #

approximateTimeDirectory(){

    GREEN='\033[0;32m'
    NC='\033[0m'

    ff=$1
    lf=$2
    dt=$3

    echo -e "${GREEN} Approximating to the nearest" $dt "value${NC}"

    # Get last and first directory or file in directory.
    firstFile_=$(ls -dvr $ff*|tail -1)
    lastFile_=$(ls -dv $lf*| tail -1)
    # Remove trailing slashes from dir name
    firstFile=$(echo $firstFile_ | sed 's:/*$::')
    lastFile=$(echo $lastFile_ | sed 's:/*$::')

    echo "  First file:" $firstFile
    echo "  Last file:" $lastFile

    for f in $(seq $ff $dt $lf); do
        #new=$(printf "$f "$f*")
        echo "$f*" "to be moved into" "$f"
        mv -i -- $f* $f
    done
    
    echo -e "${GREEN} Done.               ${NC}"    
}


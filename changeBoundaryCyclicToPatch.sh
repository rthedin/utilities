#!/bin/bash

# ------------------------------------------------------------------------------ #
# Substitute to OpenFOAM's `changeDictionary`. Such tool is deprecated and the 
# use of `foamDictionary` is now encouraged. This function gets the direction of 
# the flow and changes the boundary file information.
#
# Input: $inflowDir
# Output: changes patch information in `constant/polyMesh/boundary` from `cyclic`
#         to `patch`
# 
#
# Regis Thedin
# April 29, 2020
# regis.thedin@nrel.gov
# ------------------------------------------------------------------------------- #

changeBoundaryCyclicToPatch(){

    declare -a dir

    case $1 in
        north|south)     dir=(north south);;
        east|west)       dir=(east west);;
        *)               dir=(north south east west);;
    esac

    for d in "${dir[@]}"; do
        foamDictionary -entry entry0.$d.type      -set "patch" constant/polyMesh/boundary 
        foamDictionary -entry entry0.$d.inGroups       -remove constant/polyMesh/boundary
        foamDictionary -entry entry0.$d.matchTolerance -remove constant/polyMesh/boundary
        foamDictionary -entry entry0.$d.transform      -remove constant/polyMesh/boundary
        foamDictionary -entry entry0.$d.neighbourPatch -remove constant/polyMesh/boundary
    done

}

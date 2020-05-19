#!/bin/bash

# --------------------
# estimateCells.sh
# 
# Read domain limits and cell size from `setUp` file and give a estimate
# of number of cells for processor splitting
#
# Regis Thedin
# May 19, 2020
# -------------------------


estimateCells(){

    RED='\033[0;31m'
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color

    setup=$1

    xmin=$(foamDictionary -entry "xMin" -value $1) 
    ymin=$(foamDictionary -entry "yMin" -value $1) 
    xmax=$(foamDictionary -entry "xMax" -value $1) 
    ymax=$(foamDictionary -entry "yMax" -value $1) 
    
    zmin=$(foamDictionary -entry "zMin" -value $1) 
    zmax1=$(foamDictionary -entry "zMax1" -value $1) 
    zmin2=$(foamDictionary -entry "zMin2" -value $1) 
    zmax2=$(foamDictionary -entry "zMax2" -value $1) 
    zmin3=$(foamDictionary -entry "zMin3" -value $1) 
    zmax3=$(foamDictionary -entry "zMax3" -value $1) 
    zmin4=$(foamDictionary -entry "zMin4" -value $1) 
    zmax=$(foamDictionary -entry "zMax" -value $1) 

    nx1=$(foamDictionary -entry "nx1" -value $1) 
    nx2=$(foamDictionary -entry "nx2" -value $1) 
    nx3=$(foamDictionary -entry "nx3" -value $1) 
    nx4=$(foamDictionary -entry "nx4" -value $1) 
    ny1=$(foamDictionary -entry "ny1" -value $1) 
    ny2=$(foamDictionary -entry "ny2" -value $1) 
    ny3=$(foamDictionary -entry "ny3" -value $1) 
    ny4=$(foamDictionary -entry "ny4" -value $1) 
    nz1=$(foamDictionary -entry "nz1" -value $1) 
    nz2=$(foamDictionary -entry "nz2" -value $1) 
    nz3=$(foamDictionary -entry "nz3" -value $1) 
    nz4=$(foamDictionary -entry "nz4" -value $1) 

    x=$(echo "$xmax - $xmin" | bc)
    y=$(echo $ymax - $ymin | bc)
    z1=$(echo $zmax1 - $zmin  | bc)
    z2=$(echo $zmax2 - $zmin2 | bc)
    z3=$(echo $zmax3 - $zmin3 | bc)
    z4=$(echo $zmax - $zmin4  | bc)

    cores=$(foamDictionary -entry "nCores" -value $1) 

    if [ $zmax1 -ne $zmin2 ] || [ $zmax2 -ne $zmin3 ] || [ $zmax3 -ne $zmin4 ]; then
        echo -e "${RED}WARNING: blockMesh blocks don't stack perfectly.${NC}"
    fi

    if [ $(echo $x / $nx1|bc) -ne $(echo $y / $ny1|bc) ] || \
        [ $(echo $z1 / $nz1|bc) -ne $(echo $z1 / $nz1|bc) ]; then
        echo -e "${RED}WARNING: cells are not isotropic in level 1${NC}"
    else
        echo "Grid resolution on block 1:" $(echo $x / $nx1 |bc)
    fi
    if [ $(echo $x / $nx2|bc) -ne $(echo $y / $ny2|bc) ] || \
        [ $(echo $z2 / $nz2|bc) -ne $(echo $z2 / $nz2|bc) ]; then
        echo -e "${RED}WARNING: cells are not isotropic in level 2${NC}"
    else
        echo "Grid resolution on block 2:" $(echo $x / $nx2 |bc)
    fi
    if [ $(echo $x / $nx3|bc) -ne $(echo $y / $ny3|bc) ] || \
        [ $(echo $z3 / $nz3|bc) -ne $(echo $z3 / $nz3|bc) ]; then
        echo -e "${RED}WARNING: cells are not isotropic in level 3${NC}"
    else
        echo "Grid resolution on block 3:" $(echo $x / $nx3 |bc)
    fi
    if [ $(echo $x/$nx4|bc) -ne $(echo $y/$ny4|bc) ] || \
        [ $(echo $z4/$nz4|bc) -ne $(echo $z4/$nz4|bc) ]; then
        echo -e "${RED}WARNING: cells are not isotropic in level 4${NC}"
    else
        echo "Grid resolution on block 4:" $(echo $x / $nx4 |bc)
    fi

    cellblock1=$(echo $nx1*$ny2*$nz1 |bc)
    cellblock2=$(echo $nx2*$ny2*$nz2 | bc)
    cellblock3=$(echo $nx3*$ny3*$nz3 | bc)
    cellblock4=$(echo $nx4*$ny4*$nz4 | bc)

    celltotal=$(($cellblock1+$cellblock2+$cellblock3+$cellblock4))
    cellpercore=$(echo $celltotal/$cores |bc)

    echo -e "${GREEN}Total number of cells is${NC} $celltotal"
    echo -e "Approximately $cellpercore cells per core, using $cores cores ($(($cores/36)) nodes)"

}

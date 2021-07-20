#!/bin/bash

# ------------------------------------------------------------- #
# estimateCells.sh
# 
# Read domain limits and cell size from `setUp` file and give
# a estimate of number of cells for processor splitting
#
# Regis Thedin
# regis.thedin@nrel.gov
# May 19, 2020
# ------------------------------------------------------------- #

estimateCells(){

    RED='\033[0;31m'
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color

    if [ $# -eq 0 ]; then
        if [ -f setUp ]; then
            file=${1:-setUp}
        elif [ -f setUp.neutral ]; then
            file=${1:-setUp.neutral}
        elif [ -f setUp.unstable ]; then
            file=${1:-setUp.unstable}
        elif [ -f setUp.stable ]; then
            file=${1:-setUp.stable}
        fi
        echo -e "${RED}No 'setUp' file given.${NC} Getting grid information from '$file'"
    else
        file=$1
        if [ ! -f $file ]; then
            echo "File $file does not exist"
            return
        fi
        echo "Getting grid information from '$file'"
    fi

    xmin=$(foamDictionary -entry "xMin" -value $file) 
    ymin=$(foamDictionary -entry "yMin" -value $file) 
    xmax=$(foamDictionary -entry "xMax" -value $file) 
    ymax=$(foamDictionary -entry "yMax" -value $file) 
    
    zmin=$(foamDictionary -entry "zMin" -value $file) 
    zmax1=$(foamDictionary -entry "zMax1" -value $file) 
    zmin2=$(foamDictionary -entry "zMin2" -value $file) 
    zmax2=$(foamDictionary -entry "zMax2" -value $file) 
    zmin3=$(foamDictionary -entry "zMin3" -value $file) 
    zmax3=$(foamDictionary -entry "zMax3" -value $file) 
    zmin4=$(foamDictionary -entry "zMin4" -value $file) 
    zmax=$(foamDictionary -entry "zMax" -value $file) 

    nx1=$(foamDictionary -entry "nx1" -value $file) 
    nx2=$(foamDictionary -entry "nx2" -value $file) 
    nx3=$(foamDictionary -entry "nx3" -value $file) 
    nx4=$(foamDictionary -entry "nx4" -value $file) 
    ny1=$(foamDictionary -entry "ny1" -value $file) 
    ny2=$(foamDictionary -entry "ny2" -value $file) 
    ny3=$(foamDictionary -entry "ny3" -value $file) 
    ny4=$(foamDictionary -entry "ny4" -value $file) 
    nz1=$(foamDictionary -entry "nz1" -value $file) 
    nz2=$(foamDictionary -entry "nz2" -value $file) 
    nz3=$(foamDictionary -entry "nz3" -value $file) 
    nz4=$(foamDictionary -entry "nz4" -value $file) 

    x=$(echo "$xmax - $xmin" | bc)
    y=$(echo $ymax - $ymin | bc)
    z1=$(echo $zmax1 - $zmin  | bc)
    z2=$(echo $zmax2 - $zmin2 | bc)
    z3=$(echo $zmax3 - $zmin3 | bc)
    z4=$(echo $zmax - $zmin4  | bc)

    cores=$(foamDictionary -entry "nCores" -value $file) 

    if [ $zmax1 -ne $zmin2 ] || [ $zmax2 -ne $zmin3 ] || [ $zmax3 -ne $zmin4 ]; then
        echo -e "${RED}WARNING: blockMesh blocks don't stack perfectly.${NC}"
    fi

    if [ $(echo $x / $nx1|bc) -ne $(echo $y / $ny1|bc) ] || \
        [ $(echo $x / $nx1|bc) -ne $(echo $z1 / $nz1|bc) ]; then
        echo -e "${RED}WARNING: cells are not isotropic on block 1${NC}"
        echo "Grid resolution on block 1:" $(echo $x / $nx1 |bc) "by" $(echo $y/$ny1|bc) "by" $(echo $z1/$nz1|bc)
    else
        echo "Grid resolution on block 1:" $(echo $x / $nx1 |bc) "(uniform)"
    fi
    if [ $(echo $x / $nx2|bc) -ne $(echo $y / $ny2|bc) ] || \
        [ $(echo $x / $nx2|bc) -ne $(echo $z2 / $nz2|bc) ]; then
        echo -e "${RED}WARNING: cells are not isotropic on block 2${NC}"
        echo "Grid resolution on block 2:" $(echo $x / $nx2 |bc) "by" $(echo $y/$ny2|bc) "by" $(echo $z2/$nz2|bc)
    else
        echo "Grid resolution on block 2:" $(echo $x / $nx2 |bc) "(uniform)"
    fi
    if [ $(echo $x / $nx3|bc) -ne $(echo $y / $ny3|bc) ] || \
        [ $(echo $x / $nx3|bc) -ne $(echo $z3 / $nz3|bc) ]; then
        echo -e "${RED}WARNING: cells are not isotropic on block 3${NC}"
        echo "Grid resolution on block 3:" $(echo $x / $nx3 |bc) "by" $(echo $y/$ny3|bc) "by" $(echo $z3/$nz3|bc)
    else
        echo "Grid resolution on block 3:" $(echo $x / $nx3 |bc) "(uniform)"
    fi
    if [ $(echo $x/$nx4|bc) -ne $(echo $y/$ny4|bc) ] || \
        [ $(echo $x/$nx4|bc) -ne $(echo $z4/$nz4|bc) ]; then
        echo -e "${RED}WARNING: cells are not isotropic on block 4${NC}"
        echo "Grid resolution on block 4:" $(echo $x / $nx4 |bc) "by" $(echo $y/$ny4|bc) "by" $(echo $z4/$nz4|bc)
    else
        echo "Grid resolution on block 4:" $(echo $x / $nx4 |bc) "(uniform)"
    fi

    cellblock1=$(echo $nx1*$ny1*$nz1 |bc)
    cellblock2=$(echo $nx2*$ny2*$nz2 | bc)
    cellblock3=$(echo $nx3*$ny3*$nz3 | bc)
    cellblock4=$(echo $nx4*$ny4*$nz4 | bc)

    celltotal=$(($cellblock1+$cellblock2+$cellblock3+$cellblock4))
    cellpercore=$(echo $celltotal/$cores |bc)
    
    echo -e "${GREEN}Number of cells in block 1:${NC} $cellblock1 \t ($(( ($cellblock1*100)/$celltotal))%)"
    echo -e "${GREEN}Number of cells in block 2:${NC} $cellblock2 \t ($(( ($cellblock2*100)/$celltotal))%)"
    echo -e "${GREEN}Number of cells in block 3:${NC} $cellblock3 \t ($(( ($cellblock3*100)/$celltotal))%)"
    echo -e "${GREEN}Number of cells in block 4:${NC} $cellblock4 \t ($(( ($cellblock4*100)/$celltotal))%)"
    echo -e "${GREEN}Total number of cells is${NC} $celltotal"
    echo -e "Approximately $cellpercore cells per core, using $cores cores ($(($cores/36)) nodes)"

}

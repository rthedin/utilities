#!/bin/bash

shiftFlatBlendToz0(){
    module load conda
    conda activate myenv

    terrainSTL=${1%.*}
    x=$2
    y=$3

    echo "Shifting ${terrainSTL}.stl"
    
    z0=$(python ~/utilities/extract_elevation_from_stl.py $1 $x,$y | tail -1 | awk '{print $3}')
    z0translate=$(echo $z0*-1 | bc)

    echo "The z value at ($x,$y) is $z0"

    surfaceTransformPoints -translate '(0 0 '"$z0translate"' )' ${terrainSTL}.stl ${terrainSTL}_flatz0.stl

    echo "Before:"
    surfaceCheck ${terrainSTL}.stl | grep "Bounding"
    echo "After:"
    surfaceCheck ${terrainSTL}_flatz0.stl | grep "Bounding"
    echo "Done."

    rm -f problemFaces

   module unload conda
}

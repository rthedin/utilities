# Perform a tiled interpolated copy of a source solution onto this mesh 
tiledMapFields2D() 
{ 
    # Tiled mapping works by first taking this mesh and translating it in space to lie 
    # over the top of the source mesh, perform the mapping on the overlapping section of 
    # mesh, move the mesh again by the length of the source mesh, map, and so on until 
    # the destination mesh is completely mapped. 

    # Set variables to input arguments 
    nTileX=$1 
    nTileY=$2 
    cores=$3 
    precSizeX=$4
    precSizeY=$5 
    mapTime=$6
    sourceCaseDir=$7 
    xmin_prec=$8
    ymin_prec=$9
    xmin_case=${10}
    ymin_case=${11}
  
    # Determine the initial offset between precursors and final case origin and do initial translation
    offset_x=$((xmin_prec-xmin_case))
    offset_y=$((ymin_prec-ymin_case))
    if [ $offset_x -ne 0 ] || [ $offset_y -ne 0 ]; then 
        echo "      -transforming points by vector ($offset_x $offset_y 0)..."
        srun -n $cores transformPoints -translate "($offset_x $offset_y 0)" -parallel > log.1.transformPoints.initial 2>&1 
    fi
  
    # Tiling loop 
    for ((i=0; i<$nTileX; i++)); do
        for ((j=0; j<$nTileY; j++)); do
             echo -n "   -Performing solution mapping on tile ($i,$j): "
             echo -n "mapping precursor ($xmin_prec:$((xmin_prec+precSizeX)), $ymin_prec:$((ymin_prec+precSizeY))) into "
             echo    "($((xmin_case+i*precSizeX)):$((xmin_case+(i+1)*precSizeX)), $((ymin_case+j*precSizeY)):$((ymin_case+(j+1)*precSizeY)))..."
 
             # Perform mapping. This way is slower, but it is cleaner and actually works. Assumes we have recomposed precursor, which we do
             mapFields -mapMethod mapNearest -sourceTime $mapTime -parallelTarget $sourceCaseDir > log.1.mapFields.tile$((i))_$((j)) 2>&1

             # Let's calculate where to translate next. We need to loop over two dimension, so we need to know
             # whether or not we are on the last tile in y? If yes, the next translation need to reset the position in y.
             if [ $((j+1)) -eq $nTileY ]; then
                 tran_x=$((-1*precSizeX))
                 tran_y=$((-1*(-j*precSizeY)))
             else
                 tran_x=0
                 tran_y=$((-1*precSizeY))
             fi
  
             # Move the destination mesh such that the current "tile" overlaps the source mesh for the next loop.
             # If we are on the last loop, move it back to its original location
             if [ $((i+1)) -eq $nTileX ] && [ $((j+1)) -eq $nTileY ]; then
                 istr="($((precSizeX*i-offset_x)) $((precSizeY*j-offset_y)) 0)"
                 echo "   -Translating back to original location with vector $istr..."
                 srun -n $cores transformPoints -translate "$istr" -parallel > log.1.transformPoints.final 2>&1 
             else
                 istr="($tran_x $tran_y 0)" 
                 echo "      -transforming points by vector $istr..." 
                 srun -n $cores transformPoints -translate "$istr" -parallel > log.1.transformPoints.tile$((i))_$((j)) 2>&1 
             fi
        done 
    done
  
}

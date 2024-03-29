# Perform a tiled interpolated copy of a source solution onto this mesh 
tiledMapFields() 
{ 
   # Set variables to input arguments 
   nTranslations_=$1 
   nCoresMap_=$2 
   translationVectorX_=$3 
   translationVectorY_=$4 
   translationVectorZ_=$5 
   sourceTime_=$6 
   sourceCaseDir_=$7 
 
   # Many log files will be created.  Make a directory in which to collect them. 
   if [ ! -d logs.mapFields ] 
   then 
       mkdir logs.mapFields 
   fi 
 
   # Decompose the domain to the number of cores that will do the pseudo-parallel 
   # operation of mapFields 
   echo "   -Decomposing domain to $nCoresMap_..." 
   cd system 
   if [ -L decomposeParDict ] 
   then 
       rm decomposeParDict 
   fi 
   ln -s decomposeParDict.sub decomposeParDict 
   cd ../ 
   srun decomposePar -cellDist -force > logs.mapFields/log.decomposePar.sub 2>&1 
 
   # Tiled mapping works by first taking this mesh and translating it in space to lie 
   # over the top of the source mesh, perform the mapping on the overlapping section of 
   # mesh, move the mesh again by the length of the source mesh, map, and so on until 
   # the destination mesh is completely mapped. 
 
   # Tiling loop 
   i=0 
   while [ $i -lt $nTranslations_ ] 
   do 
      echo "   -Performing solution mapping on tile $i..." 
 
      # This part keeps track of how much total destination mesh translation has occured 
      # such that in the end, we can move the destination mesh back to its original position.      
      invtx=`echo "-1.0*($i+1)*$translationVectorX_" | bc` 
      invty=`echo "-1.0*($i+1)*$translationVectorY_" | bc` 
      invtz=`echo "-1.0*($i+1)*$translationVectorZ_" | bc` 
 
      # Move the destination mesh such that the current "tile" overlaps the source mesh. 
      echo "      -transforming points by vector ("$translationVectorX_ $translationVectorY_ $translationVectorZ_")..." 
      istr="($translationVectorX_ $translationVectorY_ $translationVectorZ_)" 
      srun -n $nCoresMap_ transformPoints -translate "$istr" -parallel > logs.mapFields/log.transformPoints.$i 2>&1 
 
      # Run mapFields to do the interpolation from source to destination mesh.  We can do this  
      # in pseudo-parallel by having a core work on each processor directory created earlier. 
      echo "      -spawning mapFields on core:" 
      j=0 
      while [ $j -lt $nCoresMap_ ] 
      do 
         echo "             $j..." 
         mapFields -case processor$j -mapMethod mapNearest -sourceTime $sourceTime_ -parallelSource $sourceCaseDir_ > logs.mapFields/log.mapFields.tile$i.core$j 2>&1 & 
         let j=j+1 
      done 
 
      # The "wait" command is there so that all mapFields jobs must finish before moving ahead. 
      echo "      -waiting for all mapFields jobs to complete..." 
      wait 
      echo "      -all mapFields jobs complete..." 
 
      let i=i+1 
   done 
 
   # Move the destination mesh back to its original position. 
   echo "   -Translating back to original with vector ("$invtx $invty $invtz ")..." 
   istr="($invtx $invty $invtz)" 
   srun -n $nCoresMap_ transformPoints -translate "$istr" -parallel > logs.mapFields/log.transformPoints.final 2>&1 
 
   # Since the mapping was done in pseudo-parallel, reconstruct the domain. 
   echo "   -Reconstructing domain..." 
   reconstructPar -time $sourceTime_ > logs.mapFields/log.reconstructPar.sub 2>&1 
}

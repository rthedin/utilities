#!/bin/bash

# ------------------------------------------------------------------- #
# Batch submission of multiple scripts at once, chaining them through 
# the `dependency` flag, with the `afterok` condition. If a single
# script is passed, a regular sbatch occurs
#
# Regis Thedin
# April 20, 2020
# regis.thedin@nrel.gov
# ------------------------------------------------------------------- #

sbatchAfterOK(){

if [ "$#" -eq 0 ]; then
    return
fi

for arg; do
  if [ ! -f $arg ]; then 
    echo $arg "does not exist."
    return;
fi
done

echo "Submitting the first job," $1
jprev=$(sbatch $1 | cut -f 4 -d ' ')

for arg in ${@:2}; do
   echo "Submitting job" $arg "depending on job" $jprev
   jnext=$(sbatch --dependency=afterok:$jprev $arg | cut -f 4 -d ' ')
   jprev=$jnext
done
 
}

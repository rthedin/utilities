#!/bin/bash
# Written by Eliot Quon

foamOK(){
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color

    check()
    {
        donestr=''
        for arg in "$@"; do
            if [ -z "$donestr" ]; then
                donestr="$arg"
            elif [ ! -f "$arg" ]; then
                echo "Current directory does not have $arg"
            else
                logfile="$arg"
                found=`grep "$donestr" $logfile`
                if [ "$?" == 0 ]; then
                    echo -e "${GREEN}$logfile${NC}: $found"
                else
                    echo -e "${RED}$logfile${NC}: '$donestr' not found ***"
                fi
            fi
        done
    }

    check 'End' log.*blockMesh*
    check 'Mesh size' log.*renumberMesh*
    check 'End' log.*renumberMesh*
    check 'End' log.*topoSet*
    check 'Refined' log.*refineHexMesh*
    check 'End' log.*refineHexMesh*
    check 'Mesh OK' log.*checkMesh*
    check 'End' log.*changeDictionary*
    check 'Max number of cells' log.*decomposePar*
    check 'End' log.*decomposePar*
    check 'Finalising' log.*moveDynamicMesh*
    check 'Finalising' log.*setFieldsABL*
    check 'Finalising' log.*superDeliciousVanilla*
}

#!/bin/bash

# written by regis thedin

pwdd(){
    apath=$(pwd -P)
    rpath=$(pwd)
    # Get the prefix (/kfs2 on kestrel; /lustra/eaglefs on eagle)
    prefixpath=$(dirname "$(readlink -f "/projects")")

    if [[ "${apath#$prefixpath}" == "$rpath" ]]; then
        pwd
    else
        echo "Relative path: $rpath"
        echo "Absolute path: ${apath#$prefixpath}"
    fi
}


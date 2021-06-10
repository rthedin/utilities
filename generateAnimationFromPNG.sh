#!/bin/bash

# ------------------------------------------------------------------- #
# Scrip to generate a video using ffmpeg from a series of pngs.
# Framerate can be given as an optional argument; default=10fps.
#
# Example call:
#       generateAnimationFromPNG 8
#       generateAnimationFromPNG
#
# Regis Thedin
# June 10, 2021
# regis.thedin@nrel.gov
# ------------------------------------------------------------------- #

generateAnimationFromPNG(){

    # Check paths
    if [[ ${PWD##*/} == animation ]]; then
        :
    elif [ -d animation ]; then
        cd animation
    else
        echo "The animation directory does not seem to exist on the top level directory of this case."
        return 1
    fi

    # get optional framerate from function call if defined
    frate=${1:-10}

    # get unique slices saved as pngs
    uniqueSlices=$(for file in *.png; do echo "${file%.*.png}"; done | sort -u)

    for slice in $uniqueSlices; do
        yes | ~/share/ffmpeg-git-20191022-amd64-static/ffmpeg -framerate $frate -i ${slice}.0%3d.png -vcodec libx264 -crf 15 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" video_${slice}.mp4
    done

}

#!/bin/bash

SCRIPT_NAME=$(basename "$0")
if [ -z $1 ]; then
    echo "Removes audio from a copy of the given video using FFMPEG utility. Doesn't change the original file."
    echo "./$SCRIPT_NAME video.file  # creates a muted_video.file (original file without audio)"
    exit 1
fi

NEW_NAME="muted_$(basename "$1")"
ffmpeg -i $1 -vcodec copy -an $NEW_NAME

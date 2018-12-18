#!/usr/bin/env bash

SCRIPT_NAME=$(basename "$0")
if [ -z $1 ] || [ -z $2 ] || [ -z $3 ]; then
    echo "Cut video using FFMPEG utility (without re-encoding). Doesn't change the original file. Creates output named 'cut_{origName}'"
    echo "Usage: ./$SCRIPT_NAME input_file start_cut_time output_length"
    echo "Example: ./$SCRIPT_NAME input.mp4 00:00:05.000 00:00:20.000"
    exit 1
fi

NEW_NAME="cut_$(basename "$1")"
ffmpeg -i $1 -vcodec copy -acodec copy -ss $2 -t $3 $NEW_NAME

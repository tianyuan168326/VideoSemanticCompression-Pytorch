#!/usr/bin/env bash

cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py diving48 /data_video/diving48/frames --num-split 1 --level 1 --subset train --format rawframes --shuffle
PYTHONPATH=. python tools/data/build_file_list.py diving48 /data_video/diving48/frames --num-split 1 --level 1 --subset val --format rawframes --shuffle
echo "Filelist for rawframes generated."

cd tools/data/diving48/

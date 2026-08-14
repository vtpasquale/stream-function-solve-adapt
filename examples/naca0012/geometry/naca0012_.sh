#! /bin/bash

# Need ESP and Refine
# source /home/vtpasquale/projects/ESP129/EngSketchPad/ESPenv.sh # (this messes up the Python settings)
PATH=/home/vtpasquale/projects/ESP129/EngSketchPad/bin:$PATH
export ESP_ROOT=/home/vtpasquale/projects/ESP129/EngSketchPad/
export LD_LIBRARY_PATH=/home/vtpasquale/projects/ESP129/EngSketchPad/lib:/home/vtpasquale/projects/ESP129/OpenCASCADE-7.8.1/lib
PATH=/home/vtpasquale/projects/refine/build/bin:$PATH

rm -rvf *.egads *.tec *.meshb *.mapbc *.jrnl

PROJECT=${0%.sh}

serveCSM -batch ${PROJECT}.csm

ref bootstrap ${PROJECT}.egads

# ln -sf ${PROJECT}-vol.meshb ${PROJECT}01.meshb

# ref adapt ${PROJECT}-vol.meshb -x ${PROJECT}-adapt2.meshb -g ${PROJECT}.egads --viscous-tags 3,4 --spalding 1e-2 1e3


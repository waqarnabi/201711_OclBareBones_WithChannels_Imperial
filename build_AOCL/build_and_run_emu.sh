#!/bin/bash -x

#Run this script to build and run AOCL emulation
#Run it with two dots like this: . ./build_and_run_emu.sh

echo -e "** Did you run the script with 2 dots like this: . ./build_and_run_emu.sh**"
echo -e "================================="
echo -e "Clean previous build and output files"
echo -e "================================="
rm -f host.exe error.log out.dat kernels_noChannels.aoco kernels_noChannels.aocx
echo -e "================================="
echo -e "Building Kernel"
echo -e "================================="
aoc -v --report -march=emulator --board p385_hpc_d5 -DTARGET=AOCL -DDEBUG ../device/kernels_noChannels.cl
echo -e "================================="
echo -e "Building Host"
echo -e "================================="
make
echo -e "================================="
echo -e "Executing"
echo -e "================================="
env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./host.exe kernels_noChannels.aocx

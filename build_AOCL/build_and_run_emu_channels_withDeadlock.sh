#!/bin/bash -x

#Run this script to build and run AOCL emulation
#Run it with two dots like this: . ./<scriptName>.sh

echo -e "** Did you run the script with 2 dots like this: . ./<scriptName>.sh **"
echo -e "================================="
echo -e "Clean previous build and output files"
echo -e "================================="
rm -f host.exe error.log out.dat kernels.aoco kernels_channels.aocx
echo -e "================================="
echo -e "Building Kernel"
echo -e "================================="
aoc -v --report -march=emulator --board p385_hpc_d5 -DCHANNELS -DTARGET=AOCL -DDEADLOCK -DDEBUG ../device/kernels_channels.cl
echo -e "================================="
echo -e "Building Host"
echo -e "================================="
make CHANNELS=1 DEADLOCK=1
echo -e "================================="
echo -e "Executing"
echo -e "================================="
env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./host.exe kernels_channels.aocx

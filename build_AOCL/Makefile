AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_LINK_CONFIG=$(shell aocl link-config)

LDLIBS      = $(AOCL_LINK_CONFIG)
CXXFLAGS    = $(AOCL_COMPILE_CONFIG) -DTARGET=AOCL -DLINUX -pthread -O2
HOST_SRCS   = ../host/main.cpp
CXX         = /usr/bin/g++
HOST_EXE		= host.exe
OUTPUTFILES = error.log out.dat

ifdef CHANNELS
CXXFLAGS += -DCHANNELS
endif

host: $(HOST_EXE)

$(HOST_EXE): $(HOST_SRCS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
	
clean: 
	rm -f $(HOST_EXE) $(OUTPUTFILES)

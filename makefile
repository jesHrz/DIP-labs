ifeq ($(V),1)
override V =
endif
ifeq ($(V),0)
override V = @
endif

CC = g++

OPENCV_CCFLAGS 	  := $(shell pkg-config --cflags opencv)
OPENCV_LIBS 	  := $(shell pkg-config --libs opencv)

CCFLAGS = -g -Wall $(OPENCV_CCFLAGS) -Iinc -I./ -std=c++11
LDFLAGS = -Wl, $(OPENCV_LIBS)

OBJDIR 	:= obj
OBJDIRS := 

all:

include lab1/Makefrag

.PHONY : clean run
clean:
	-rm -rf $(OBJDIR)

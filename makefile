ifeq ($(V),1)
override V =
endif
ifeq ($(V),0)
override V = @
endif

CC = g++

OPENCV_CCFLAGS 	  := $(shell pkg-config --cflags opencv)
OPENCV_LIBS 	  := $(shell pkg-config --libs opencv)

CCFLAGS = -g -Wall $(OPENCV_CCFLAGS) -I./ -std=c++11
LDFLAGS = -Wl, $(OPENCV_LIBS)

OBJDIR 	:= obj
OBJDIRS := 

all:

include lab1/Makefrag
include lab2/Makefrag
include lab3/Makefrag
include lab4/Makefrag
include lab5/Makefrag

include sidewindow/Makefrag

.PHONY : clean
clean:
	-rm -rf $(OBJDIR)

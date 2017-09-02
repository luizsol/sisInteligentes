# Makefile

# Compiler
CC = g++

# Compiler flags
CFLAGS = -std=c++11

# Enable warnings
CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wpedantic

# Library includes
LIBS = -lopencv_core -lopencv_highgui

# Directories
TDIR = test

# Goals
.DEFAULT_GOAL := main

# Object files
Main.o: Main.cpp
	$(CC) $(CFLAGS) -c Main.cpp

# All
main: Main.o
	$(CC) $(CFLAGS) Main.o -o main $(LIBS)

clean:
	rm -f *.o main
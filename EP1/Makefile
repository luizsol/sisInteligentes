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
LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video
LIBS += -lopencv_objdetect

# Directories
TDIR = test
SRC = src

# Goals
.DEFAULT_GOAL := main

# Object files
ImgProcessor.o: $(SRC)/ImgProcessor.cpp $(SRC)/ImgProcessor.h
	$(CC) $(CFLAGS) -c $(SRC)/ImgProcessor.cpp

Main.o: $(SRC)/Main.cpp
	$(CC) $(CFLAGS) -c $(SRC)/Main.cpp

# All
main: Main.o ImgProcessor.o
	$(CC) $(CFLAGS) Main.o ImgProcessor.o -o main $(LIBS)

clean:
	rm -f *.o main
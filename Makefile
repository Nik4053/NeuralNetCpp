CC = g++
CCFLAGS = -g -std=c++17 -O3 -march=native
TARGET = mnistTest

.PHONY: all clean

all: $(TARGET)

clean:
	rm -rf *.o $(TARGET)

$(TARGET): $(TARGET).o
	$(CC) $^  $(CCFLAGS) -o $@ 
	
$(TARGET).o: $(TARGET).cpp
	$(CC) $^ $(CCFLAGS) -c  $@
	
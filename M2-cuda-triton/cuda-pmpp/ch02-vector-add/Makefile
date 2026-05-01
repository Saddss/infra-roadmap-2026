NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_70

TARGET = vector_add
SRCS = kernel.cu main.cu

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean

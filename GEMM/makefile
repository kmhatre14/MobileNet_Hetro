### OpenCL “matrixmul" Makefile
### Assumes preferred CUDA environment module is loaded
CC  = gcc
CFLAGS  = -I${CUDA_INSTALL_PATH}/include -L${CUDA_INSTALL_PATH}/lib64 -lOpenCL
TARGET  = conMat
default: $(TARGET)
$(TARGET):
	$(CC) $(CFLAGS) -o $(TARGET) matrixmul_host.c -lm
clean:
	rm $(TARGET)

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>

#define W 224
#define H 224
#define K_D 3
#define CHANNELS 3

/*Function Prototype*/
int decode_image(unsigned char* frame, char filename[]);
void stitchChannels(unsigned char* imd,unsigned char* imOut);
void im2col_cpu(unsigned char* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, unsigned char* data_col);
unsigned char im2col_get_pixel(unsigned char *im, int height, int width, int channels, int row, int col, int channel, int pad);
double kernelExecTimeNs;
void readSquezeNetKernel(int *m);
long LoadOpenCLKernel(char const* path, char **buf);
int openCldeviceConfig( void );
int openCLContextConfig( void );

/*Cl device Variables */
int err;                            // error code returned from api calls
cl_device_id device_id;             // compute device id 
cl_context context;                 // compute context
cl_command_queue commands;          // compute command queue
cl_program program;                 // compute program
cl_kernel kernel;                   // compute kernel
cl_event myevent;
cl_ulong start;
cl_ulong end;
cl_uint dev_cnt = 0;
cl_platform_id platform_ids[100];

// OpenCL device memory for matrices
cl_mem d_image;
cl_mem d_filter;
cl_mem d_C;
char *KernelSource;
long lFileSize;
size_t localWorkSize[2], globalWorkSize[2];

/*Application variables*/
int imgcount = 0;
char count_buff[5];
char filebuff[50];
int colSize;
unsigned int mem_size_C;
unsigned int mem_size_op_im2col;
unsigned char* h_op_im2col;
unsigned int mem_size_filter;
unsigned int mem_size_image;
unsigned char* h_filter;
unsigned int size_image;
unsigned char* h_image;
unsigned int size_filter;
unsigned int size_op_im2col;
int dG_h,dG_w;
unsigned char* h_imStitchChannel;
unsigned int size_C;
unsigned char* h_C;

int main(int argc, char** argv)
{
    // set seed for rand()
    srand(2014);
   
    //Allocate host memory for image with 3 channels
    size_image = W * H * K_D;
    mem_size_image = sizeof(unsigned char) * size_image;
    h_image = (unsigned char*) malloc(mem_size_image);

    //Stitch the image such that CH1 CH2 CH3 in series
    h_imStitchChannel = (unsigned char*) malloc(mem_size_image);
    
    //Allocate host memory for filter
    size_filter = K_D * K_D * CHANNELS;
    mem_size_filter = sizeof(unsigned char) * size_filter;
    h_filter = (unsigned char*)malloc(mem_size_filter);

    //Allocate host memory for im2col matrix
    size_op_im2col = K_D*K_D*H*W*CHANNELS;
    mem_size_op_im2col = sizeof(unsigned char) * size_op_im2col;
    h_op_im2col = (unsigned char*) malloc(mem_size_op_im2col);      
    
    openCldeviceConfig();

    printf("Initializing OpenCL device...\n");
 


    decode_image(h_image,"data/dog.ppm");
    printf("Reading host image..Done\n");
    int i,j;   

    //Set filter element 0 to 1
    //readSquezeNetKernel(h_filter);
    for(i=0;i<27;i++)
    {
        h_filter[i] = 1;    
    }
    

    stitchChannels(h_image,h_imStitchChannel);

    for(i=0;i<5;i++)
    {
        for(j=0;j<5;j++)
        { 
           printf("%d \t" , h_imStitchChannel[i*W*CHANNELS+j]);
        }
        printf("\n");
    }
    printf("im2col output \n");

    im2col_cpu(h_imStitchChannel,3,H,W,K_D,1,0,h_op_im2col);
    printf("Input Image Dim H %d \t W %d \t K_D %d\n",H,W,K_D);
    printf("im2Col Image Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D*K_D*CHANNELS),(dG_h*dG_w));
    for(i=0;i<27;i++)
    {
        for(j=0;j<9;j++)
        { 
            printf("%d \t" , h_op_im2col[i*dG_h*dG_w + j]);
        }
        printf("\n");
    }

    //Allocate host memory for the result C
    size_C = dG_h * dG_w;
    mem_size_C = sizeof(unsigned char) * size_C;
    h_C = (unsigned char*) malloc(mem_size_C);

    openCLContextConfig();

    printf("Running matrix multiplication for matrices im2Col_Matrix (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D*K_D*CHANNELS),(dG_h*dG_w),1,(K_D*K_D*CHANNELS)); 

    //Launch OpenCL kernel
    int argK = K_D;
    int argH = dG_h;
    int argW = dG_w;
    int argChannel = CHANNELS;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argK);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argH);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argW);
    err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argChannel);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    //set the local and globar work group size 
    localWorkSize[0] = 2;
    localWorkSize[1] = 2;
    globalWorkSize[0] = dG_w;
    globalWorkSize[1] = dG_h;

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);
    clFinish(commands);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    kernelExecTimeNs += end-start;
    //kernelExecTimeNs += kernelExecTimeNs; 
    printf("time in nanossec %0.3f \n", kernelExecTimeNs);
    //Retrieve result from device
    err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
    printf("Error: Failed to read output array! %d\n", err);
    exit(1);
    }

    printf("Matrix multiplication completed...\n"); 
    for(i=0;i<5;i++)
    {
        for(j=0;j<5;j++)
        { 
           printf("%d \t" , h_C[i*dG_w+j]);
        }
        printf("\n");
    }
    printf("last value %d \n" , h_C[(dG_w*dG_h)-1]);
    //Shutdown and cleanup
    free(h_image);
    free(h_filter);
    free(h_C);
    free(h_op_im2col);


    clReleaseMemObject(d_image);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_C);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    return 0;
}

int decode_image(unsigned char* frame, char filename[]) {
    FILE *pFile;
    pFile = fopen(filename, "r");
    if(pFile == NULL){
        fprintf(stderr, "Could not open %s.\n", filename);
        return -1; 
    }
    fseek(pFile, 15, SEEK_SET);
    fread(frame, sizeof(char), H*W*3, pFile);   
    fclose(pFile);
    return 0; 
}
void seperateChannels(unsigned char* imd,unsigned char* im1,unsigned char* im2,unsigned char* im3){
    int i,j;    
    for(i=0,j=0; i<H*W; i++,j+=3){
        im1[i] = imd[j];
        im2[i] = imd[j+1];
        im3[i] = imd[j+2];                
    }
}
void im2col_cpu(unsigned char* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, unsigned char* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    dG_h = height_col;
    dG_w = width_col;
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
} 

unsigned char im2col_get_pixel(unsigned char *im, int height, int width, int channels, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||row >= height || col >= width)   return 0;
    return im[col + width*(row + height*channel)];
}
void readSquezeNetKernel(int *m) 
{

	FILE *fp;	
   	char buff[255];
	double n;
   	fp = fopen("snweights-ints3.txt", "r");
	int sizeInt = 7*7*3*96*sizeof(int);
	int i=0;
	for(i=1;i<7*7*3*96+1;i++)
	{	
		fscanf(fp, "%s", buff);
		n = atof(buff);
		m[i-1]=n;
	}
   fclose(fp);
}

int openCldeviceConfig( void )
{
    clGetPlatformIDs(0, 0, &dev_cnt);

    
    clGetPlatformIDs(dev_cnt, platform_ids, NULL);

    // Connect to a compute device
    int gpu = 1;
    err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

}

int openCLContextConfig( void )
{
    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source file


    lFileSize = LoadOpenCLKernel("matrixmul_kernel.cl", &KernelSource);
    if( lFileSize < 0L ) {
        perror("File read failed");
        return 1;
    }

    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "matrixMul", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_C, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_op_im2col, h_op_im2col, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter, h_filter, &err);

    if (!d_image || !d_filter || !d_C)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
}
// Allocates a matrix with random float entries.
void randomMemInit(unsigned char* data, int size)
{
   int i;

   for (i = 0; i < size; ++i)
   	data[i] = rand();
}

long LoadOpenCLKernel(char const* path, char **buf)
{
    FILE  *fp;
    size_t fsz;
    long   off_end;
    int    rc;

    /* Open the file */
    fp = fopen(path, "r");
    if( NULL == fp ) {
        return -1L;
    }

    /* Seek to the end of the file */
    rc = fseek(fp, 0L, SEEK_END);
    if( 0 != rc ) {
        return -1L;
    }

    /* Byte offset to the end of the file (size) */
    if( 0 > (off_end = ftell(fp)) ) {
        return -1L;
    }
    fsz = (size_t)off_end;

    /* Allocate a buffer to hold the whole file */
    *buf = (char *) malloc( fsz+1);
    if( NULL == *buf ) {
        return -1L;
    }

    /* Rewind file pointer to start of file */
    rewind(fp);

    /* Slurp file into buffer */
    if( fsz != fread(*buf, 1, fsz, fp) ) {
        free(*buf);
        return -1L;
    }

    /* Close the file */
    if( EOF == fclose(fp) ) {
        free(*buf);
        return -1L;
    }


    /* Make sure the buffer is NUL-terminated, just in case */
    (*buf)[fsz] = '\0';

    /* Return the file size */
    return (long)fsz;
}

void stitchChannels(unsigned char* imd,unsigned char* imOut)
{
    int i,j;    
    for(i=0,j=0; i<H*W; i++,j+=3){
        imOut[i] = imd[j];
        imOut[i+(H*W)] = imd[j+1];
        imOut[i+(H*W*2)] = imd[j+2];                
    }
}



// Added contributor shafaan
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

/*Layer1 Varaible and constants defination*/
#define W_1 224
#define H_1 224
#define K_D_1 3
#define CHANNELS_1 3 //Channels of input
#define NO_OF_FILTERS_1 32 //Channels of input
/*Layer2 Depthwise*/
#define W_2 112
#define H_2 112
#define K_D_2 3
#define CHANNELS_2 1
#define NO_OF_FILTERS_2 32 //Channels of input
/*Layer3 Pointwise*/
#define W_3 112
#define H_3 112
#define K_D_3 1
#define CHANNELS_3 32
#define NO_OF_FILTERS_3 64 //Channels of input
/*Layer4 Depthwise*/
#define W_4 112
#define H_4 112
#define K_D_4 3
#define CHANNELS_4 64
#define NO_OF_FILTERS_4 64 //Channels of input
/*Layer5 Pointwise*/
#define W_5 56
#define H_5 56
#define K_D_5 1
#define CHANNELS_5 64
#define NO_OF_FILTERS_5 128 //Channels of input
/*Layer6 Depthwise*/
#define W_6 56
#define H_6 56
#define K_D_6 3
#define CHANNELS_6 128
#define NO_OF_FILTERS_6 128 //Channels of input
/*Layer7 Pointwise*/
#define W_7 56
#define H_7 56
#define K_D_7 1
#define CHANNELS_7 128
#define NO_OF_FILTERS_7 128 //Channels of input
/*Layer8 Depthwise*/
#define W_8 56
#define H_8 56
#define K_D_8 3
#define CHANNELS_8 128
#define NO_OF_FILTERS_8 128 //Channels of input
/*Layer9 Pointwise*/
#define W_9 28
#define H_9 28
#define K_D_9 1
#define CHANNELS_9 128
#define NO_OF_FILTERS_9 256 //Channels of input
/*Layer10 Depthwise*/
#define W_10 28
#define H_10 28
#define K_D_10 3
#define CHANNELS_10 256
#define NO_OF_FILTERS_10 256 //Channels of input
/*Function Prototype*/
int decode_image(unsigned char* frame, char filename[]);
void stitchChannels(unsigned char* imd,unsigned char* imOut);
void im2col_cpu(unsigned char* data_im,
int channels,  int height,  int width,
int ksize,  int stride, int pad, unsigned char* data_col);
unsigned char im2col_get_pixel(unsigned char *im, int height, int width, int channels, int row, int col, int channel, int pad);
double kernelExecTimeNs;
double TotalTime;
void readSquezeNetKernel(int *m);
long LoadOpenCLKernel(char const* path, char **buf);
int openCldeviceConfig( void );
int openCLContextConfig( void );
void Layer1( void );
void Layer2( void );
void Layer3( void );
void Layer4( void );
void Layer5( void );
void Layer6( void );
void Layer7( void );
void Layer8( void );
void Layer9( void );
void Layer10( void );
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
cl_mem d_output;
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
unsigned char* im2colL1;
unsigned int mem_size_filter;
unsigned int mem_size_image;
unsigned char* filterL1;
unsigned int size_image;
unsigned char* main_image;
unsigned int size_filter;
unsigned int size_op_im2col;
int dG_h,dG_w;
unsigned char* main_image_ss;//ss stands for stitch channel
unsigned int size_C;
unsigned char* outputL1;
//Data for OP of 1st Layer
unsigned int size_l1_out;
unsigned int mem_size_l1_out;
unsigned char* outL1;

unsigned int size_l1_im2col;
unsigned int mem_size_l1_im2col;
unsigned char* im2colL2;

unsigned int size_op_l1;
unsigned int mem_size_op_l1;
unsigned char* outputL2_eachFilter;

unsigned int size_filter_l1_d;
unsigned int mem_size_filter_l1_d;
unsigned char* filterL2;
//Data for OP of Depthwise Layer
unsigned int size_l2_out;
unsigned int mem_size_l2_out;
unsigned char* outputL2;

//Defination for pointwise convolution
unsigned int size_filter_l1_p;
unsigned int mem_size_filter_l1_p;
unsigned char* filter3;

unsigned int size_l2_im2col;
unsigned int mem_size_l2_im2col;
unsigned char* im2col3;

unsigned int size_l2_out_p;
unsigned int mem_size_l2_out_p;
unsigned char* output3;

unsigned int size_op_l2;
unsigned int mem_size_op_l2;
unsigned char* h_op_l2;
//For Layer 4 Depthwise
unsigned int size_l4_im2col;
unsigned int mem_size_l4_im2col;
unsigned char* im2colL4;

unsigned int size_l4_out;
unsigned int mem_size_l4_out;
unsigned char* outputL4;

unsigned int size_filter_l4_d;
unsigned int mem_size_filter_l4_d;
unsigned char* filterL4;

unsigned int size_op_l4;
unsigned int mem_size_op_l4;
unsigned char* outputL4_eachFilter;

//For layer 5 pointwise
unsigned int size_l5_im2col;
unsigned int mem_size_l5_im2col;
unsigned char* im2col5;

unsigned int size_l5_out_p;
unsigned int mem_size_l5_out_p;
unsigned char* output5;

unsigned int size_filter_l5_p;
unsigned int mem_size_filter_l5_p;
unsigned char* filter5;

//For Layer 6 Depthwise
unsigned int size_l6_im2col;
unsigned int mem_size_l6_im2col;
unsigned char* im2colL6;

unsigned int size_l6_out;
unsigned int mem_size_l6_out;
unsigned char* outputL6;

unsigned int size_filter_l6_d;
unsigned int mem_size_filter_l6_d;
unsigned char* filterL6;

unsigned int size_op_l6;
unsigned int mem_size_op_l6;
unsigned char* outputL6_eachFilter;

//For layer 7 pointwise
unsigned int size_l7_im2col;
unsigned int mem_size_l7_im2col;
unsigned char* im2col7;

unsigned int size_l7_out_p;
unsigned int mem_size_l7_out_p;
unsigned char* output7;

unsigned int size_filter_l7_p;
unsigned int mem_size_filter_l7_p;
unsigned char* filter7;

//For Layer 8 Depthwise
unsigned int size_l8_im2col;
unsigned int mem_size_l8_im2col;
unsigned char* im2colL8;

unsigned int size_l8_out;
unsigned int mem_size_l8_out;
unsigned char* outputL8;

unsigned int size_filter_l8_d;
unsigned int mem_size_filter_l8_d;
unsigned char* filterL8;

unsigned int size_op_l8;
unsigned int mem_size_op_l8;
unsigned char* outputL8_eachFilter;

//For layer 9 pointwise
unsigned int size_l9_im2col;
unsigned int mem_size_l9_im2col;
unsigned char* im2col9;

unsigned int size_l9_out_p;
unsigned int mem_size_l9_out_p;
unsigned char* output9;

unsigned int size_filter_l9_p;
unsigned int mem_size_filter_l9_p;
unsigned char* filter9;

//For Layer 10 Depthwise
unsigned int size_l10_im2col;
unsigned int mem_size_l10_im2col;
unsigned char* im2colL10;

unsigned int size_l10_out;
unsigned int mem_size_l10_out;
unsigned char* outputL10;

unsigned int size_filter_l10_d;
unsigned int mem_size_filter_l10_d;
unsigned char* filterL10;

unsigned int size_op_l10;
unsigned int mem_size_op_l10;
unsigned char* outputL10_eachFilter;

int main(int argc, char** argv)
{
    // set seed for rand()
    srand(2014);
    openCldeviceConfig();
    /********************Layer1********************/
    //Allocate host memory for image with 3 channels
    size_image = W_1 * H_1 * K_D_1;
    mem_size_image = sizeof(unsigned char) * size_image;
    main_image = (unsigned char*) malloc(mem_size_image);

    //Stitch the image such that CH1 CH2 CH3 in series
    main_image_ss = (unsigned char*) malloc(mem_size_image);
    
    //Allocate host memory for filter
    size_filter = K_D_1 * K_D_1 * CHANNELS_1 * NO_OF_FILTERS_1;
    mem_size_filter = sizeof(unsigned char) * size_filter;
    filterL1 = (unsigned char*)malloc(mem_size_filter);

    //Allocate host memory for im2col matrix
    size_op_im2col = K_D_1*K_D_1*H_1*W_1*CHANNELS_1;
    mem_size_op_im2col = sizeof(unsigned char) * size_op_im2col;
    im2colL1 = (unsigned char*) malloc(mem_size_op_im2col); 
        
    //Allocate host memory op of 1st layer
    size_l1_out = H_1*W_1*CHANNELS_1;
    mem_size_l1_out = sizeof(unsigned char) * size_l1_out;
    outL1 = (unsigned char*) malloc(mem_size_l1_out);      
       
    printf("Initializing OpenCL device...\n");

    decode_image(main_image,"data/dog.ppm");
    printf("Reading host image..Done\n");
    /*Layer 1 : Convolution*/
    Layer1();
    /*******************Layer 1 Ends****************/

    /*******************Layer 2 Depthwise****************/

    //Allocate host memory op of 1st layer im2col
    size_l1_im2col = K_D_2*K_D_2*H_2*W_2;
    mem_size_l1_im2col = sizeof(unsigned char) * size_l1_im2col;
    im2colL2 = (unsigned char*) malloc(mem_size_l1_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l2_out = H_2*W_2*NO_OF_FILTERS_2;
    mem_size_l2_out = sizeof(unsigned char) * size_l2_out;
    outputL2 = (unsigned char*) malloc(mem_size_l2_out);      
    
    Layer2();

    /*******************Layer 2 Ends*********************/

    /*******************Layer 3 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l2_im2col = K_D_3*K_D_3*H_3*W_3*CHANNELS_3;
    mem_size_l2_im2col = sizeof(unsigned char) * size_l2_im2col;
    im2col3 = (unsigned char*) malloc(mem_size_l2_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l2_out_p = H_3*W_3*NO_OF_FILTERS_3;
    mem_size_l2_out_p = sizeof(unsigned char) * size_l2_out_p;
    output3 = (unsigned char*) malloc(mem_size_l2_out_p);      

    //Allocate host memory for filter
    size_filter_l1_p = K_D_3 * K_D_3 * CHANNELS_3 *NO_OF_FILTERS_3 ;
    mem_size_filter_l1_p = sizeof(unsigned char) * size_filter_l1_p;
    filter3 = (unsigned char*)malloc(mem_size_filter_l1_p);

    Layer3();
    /*******************Layer 3 Ends****************/
    
    /*******************Layer 4 Depthwise****************/

    //Allocate host memory op of 1st layer im2col
    size_l4_im2col = K_D_4*K_D_4*H_4*W_4;
    mem_size_l4_im2col = sizeof(unsigned char) * size_l4_im2col;
    im2colL4 = (unsigned char*) malloc(mem_size_l4_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l4_out = H_4*W_4*NO_OF_FILTERS_4;
    mem_size_l4_out = sizeof(unsigned char) * size_l4_out;
    outputL4 = (unsigned char*) malloc(mem_size_l4_out);      
    
    Layer4();

    /*******************Layer 4 Ends*********************/
    /*******************Layer 5 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l5_im2col = K_D_5*K_D_5*H_5*W_5*CHANNELS_5;
    mem_size_l5_im2col = sizeof(unsigned char) * size_l2_im2col;
    im2col5 = (unsigned char*) malloc(mem_size_l2_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l5_out_p = H_3*W_3*NO_OF_FILTERS_3;
    mem_size_l5_out_p = sizeof(unsigned char) * size_l5_out_p;
    output5 = (unsigned char*) malloc(mem_size_l5_out_p);      

    //Allocate host memory for filter
    size_filter_l5_p = K_D_5 * K_D_5 * CHANNELS_5 *NO_OF_FILTERS_5 ;
    mem_size_filter_l5_p = sizeof(unsigned char) * size_filter_l5_p;
    filter5 = (unsigned char*)malloc(mem_size_filter_l5_p);

    Layer5();
    /*******************Layer 5 Ends**********************/
    
    /*******************Layer 6 Depthwise****************/

    //Allocate host memory op of 1st layer im2col
    size_l6_im2col = K_D_6*K_D_6*H_6*W_6;
    mem_size_l6_im2col = sizeof(unsigned char) * size_l6_im2col;
    im2colL6 = (unsigned char*) malloc(mem_size_l6_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l6_out = H_6*W_6*NO_OF_FILTERS_6;
    mem_size_l6_out = sizeof(unsigned char) * size_l6_out;
    outputL6 = (unsigned char*) malloc(mem_size_l6_out);      
    
    Layer6();

    /*******************Layer 6 Ends*********************/
    
    /*******************Layer 7 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l7_im2col = K_D_7*K_D_7*H_7*W_7*CHANNELS_7;
    mem_size_l7_im2col = sizeof(unsigned char) * size_l7_im2col;
    im2col7 = (unsigned char*) malloc(mem_size_l7_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l7_out_p = H_7*W_7*NO_OF_FILTERS_7;
    mem_size_l7_out_p = sizeof(unsigned char) * size_l7_out_p;
    output7 = (unsigned char*) malloc(mem_size_l7_out_p);      

    //Allocate host memory for filter
    size_filter_l7_p = K_D_7 * K_D_7 * CHANNELS_7 *NO_OF_FILTERS_7 ;
    mem_size_filter_l7_p = sizeof(unsigned char) * size_filter_l7_p;
    filter7 = (unsigned char*)malloc(mem_size_filter_l7_p);

    Layer7();
   /*******************Layer 7 Ends****************/
   
   /*******************Layer 8 Depthwise s2****************/

    //Allocate host memory op of 1st layer im2col
    size_l8_im2col = K_D_8*K_D_8*H_8*W_8;
    mem_size_l8_im2col = sizeof(unsigned char) * size_l8_im2col;
    im2colL8 = (unsigned char*) malloc(mem_size_l8_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l8_out = H_8*W_8*NO_OF_FILTERS_8;
    mem_size_l8_out = sizeof(unsigned char) * size_l8_out;
    outputL8 = (unsigned char*) malloc(mem_size_l8_out);      
    
    Layer8();

   /*******************Layer 8 Ends*********************/
   
   /*******************Layer 9 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l9_im2col = K_D_9*K_D_9*H_9*W_9*CHANNELS_9;
    mem_size_l9_im2col = sizeof(unsigned char) * size_l9_im2col;
    im2col9 = (unsigned char*) malloc(mem_size_l9_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l9_out_p = H_9*W_9*NO_OF_FILTERS_9;
    mem_size_l9_out_p = sizeof(unsigned char) * size_l9_out_p;
    output9 = (unsigned char*) malloc(mem_size_l9_out_p);      

    //Allocate host memory for filter
    size_filter_l9_p = K_D_9 * K_D_9 * CHANNELS_9 *NO_OF_FILTERS_9 ;
    mem_size_filter_l9_p = sizeof(unsigned char) * size_filter_l9_p;
    filter9 = (unsigned char*)malloc(mem_size_filter_l9_p);

    Layer9();
   /*******************Layer 9 Ends****************/
   
  /*******************Layer 10 Depthwise s1****************/

    //Allocate host memory op of 1st layer im2col
    size_l10_im2col = K_D_10*K_D_10*H_10*W_10;
    mem_size_l10_im2col = sizeof(unsigned char) * size_l10_im2col;
    im2colL10 = (unsigned char*) malloc(mem_size_l10_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l10_out = H_10*W_10*NO_OF_FILTERS_10;
    mem_size_l10_out = sizeof(unsigned char) * size_l10_out;
    outputL10 = (unsigned char*) malloc(mem_size_l10_out);      
    
    //Layer10();

  /*******************Layer 10 Ends*********************/

    
    //Shutdown and cleanup
    free(main_image);
    free(filterL1);
    free(outputL1);
    free(im2colL1);


    clReleaseMemObject(d_image);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_output);

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
    fread(frame, sizeof(char), H_1*W_1*3, pFile);   
    fclose(pFile);
    return 0; 
}
void seperateChannels(unsigned char* imd,unsigned char* im1,unsigned char* im2,unsigned char* im3){
    int i,j;    
    for(i=0,j=0; i<H_1*W_1; i++,j+=3){
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
    kernel = clCreateKernel(program, "matrixMul", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
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
    for(i=0,j=0; i<H_1*W_1; i++,j+=3){
        imOut[i] = imd[j];
        imOut[i+(H_1*W_1)] = imd[j+1];
        imOut[i+(H_1*W_1*2)] = imd[j+2];                
    }
}

void Layer1( void )
{
    int itr,i,j,jf=0;
    stitchChannels(main_image,main_image_ss);

//    for(i=0;i<5;i++)
//    {
//        for(j=0;j<5;j++)
//           printf("%d \t" , main_image_ss[i*W_1*CHANNELS_1+j]);
//       }
//        printf("\n");
//    }
    printf("im2col output \n");

    im2col_cpu(main_image_ss,3,H_1,W_1,K_D_1,2,1,im2colL1);
    printf("im2Col Image Dim H_1 %d \t W_1 %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H_1 %d \t W_1 %d \n",(K_D_1*K_D_1*CHANNELS_1),(dG_h*dG_w));
//    for(i=0;i<27;i++)
//    {
//        for(j=0;j<9;j++)
//        { 
//            printf("%d \t" , im2colL1[i*dG_h*dG_w + j]);
//        }
//        printf("\n");
//    }

//        { 
    printf("Input Image Dim H_1 %d \t W_1 %d \t K_D_1 %d\n",H_1,W_1,K_D_1);
    //Allocate host memory for the result C
    size_C = dG_h * dG_w * NO_OF_FILTERS_1;
    mem_size_C = sizeof(unsigned char) * size_C;
    outputL1 = (unsigned char*) malloc(mem_size_C);

    openCLContextConfig();


    filterL1[0] = 0;    

    // Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_C, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_op_im2col, im2colL1, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter, filterL1, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_1;
    int argF_W = K_D_1*K_D_1*CHANNELS_1;
    int argI_H = K_D_1*K_D_1*CHANNELS_1;
    int argI_W = dG_w*dG_h*CHANNELS_1;
    int argO_W = dG_w*dG_h;
    printf("Running matrix multiplication for matrices im2Col_Matrix (%dx%d) and Filter_Matrix (%dx%d) ...\n",argI_H,argI_W,argF_H,argF_W); 

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argF_H);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argF_W);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argI_H);
    err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argI_W);
    err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&argO_W);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    //set the local and globar work group size 
    localWorkSize[0] = 2;
    localWorkSize[1] = 2;
    globalWorkSize[0] = dG_w*dG_h;
    globalWorkSize[1] = NO_OF_FILTERS_1;

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);

    clFinish(commands);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    TotalTime +=end-start;
    kernelExecTimeNs = end-start;

    printf("Layer1 time %0.3f nanossec\n", kernelExecTimeNs);
    //Retrieve result from device
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_C, outputL1, 0, NULL, NULL);
    clFinish(commands);
   
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 
    
    printf("last value %d \n" , outputL1[(dG_w*dG_h*NO_OF_FILTERS_1)-1]);
}

void Layer2( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l1_d = K_D_2 * K_D_2;
    mem_size_filter_l1_d = sizeof(unsigned char) * size_filter_l1_d;
    filterL2 = (unsigned char*)malloc(mem_size_filter_l1_d);
    kernelExecTimeNs =0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL1,1,H_2,W_2,K_D_2,1,1,im2colL2);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l1 = dG_h * dG_w;
        mem_size_op_l1 = sizeof(unsigned char) * size_op_l1;
        outputL2_eachFilter = (unsigned char*) malloc(mem_size_op_l1);


        //Create the Filter for 3x3 convolution 
        filterL2[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l1, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l1_im2col, im2colL2, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l1_d, filterL2, &err);

        if (!d_image || !d_filter || !d_output)
        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = NO_OF_FILTERS_2;
        int argF_W = K_D_2*K_D_2*CHANNELS_2;
        int argI_H = K_D_2*K_D_2*CHANNELS_2;
        int argI_W = dG_w*dG_h*CHANNELS_2;
        int argO_W = dG_w*dG_h;
        //printf("Running GEMM Layer2Depth (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_2*K_D_2),(dG_h*dG_w),1,(K_D_2*K_D_2)); 

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
        err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argF_H);
        err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argF_W);
        err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argI_H);
        err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argI_W);
        err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&argO_W);

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
        TotalTime +=end-start;

        //Retrieve result from device
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l1, outputL2_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array
        for(i=0; i<size_op_l1; i++,jf++)
        {    
            outputL2[jf] = outputL2_eachFilter[i];
        }
        
    }
    printf("Layer2 time in %0.3f nanossec\n", kernelExecTimeNs);
    printf("Depthwise Layer done %d\n",itr);
}
//point wise layer
void Layer3( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL2,CHANNELS_3,H_3,W_3,K_D_3,1,0,im2col3);

    printf("Feature Map Dim H_3 %d \t W_3 %d \t K_D_3 %d\n",H_3,W_3,K_D_3);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_3*K_D_3),(dG_h*dG_w));

    filter3[0]=1;

    printf("Running GEMM Layer2Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_3*K_D_3*CHANNELS_3),(dG_h*dG_w),1,(K_D_3*K_D_3*CHANNELS_3)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l2_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l2_im2col, im2col3, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l1_p, filter3, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_3;
    int argF_W = K_D_3*K_D_3*CHANNELS_3;
    int argI_H = K_D_3*K_D_3*CHANNELS_3;
    int argI_W = dG_w*dG_h;
    int argO_W = dG_w*dG_h;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argF_H);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argF_W);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argI_H);
    err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argI_W);
    err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&argO_W);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    //set the local and globar work group size 
    localWorkSize[0] = 2;
    localWorkSize[1] = 2;
    globalWorkSize[0] = dG_w*dG_h;
    globalWorkSize[1] = NO_OF_FILTERS_3;

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);
    clFinish(commands);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    TotalTime +=end-start;
    kernelExecTimeNs = end-start;

    printf("time in nanossec %0.3f \n", kernelExecTimeNs);
    //Retrieve result from device
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l2_out_p, output3, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer done %d\n",itr);
}
//Layre 4 depthwise s2 
void Layer4( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l4_d = K_D_4 * K_D_4;
    mem_size_filter_l4_d = sizeof(unsigned char) * size_filter_l4_d;
    filterL4 = (unsigned char*)malloc(mem_size_filter_l4_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL4,1,H_4,W_4,K_D_4,2,1,im2colL4);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l4 = dG_h * dG_w;
        mem_size_op_l4 = sizeof(unsigned char) * size_op_l4;
        outputL4_eachFilter = (unsigned char*) malloc(mem_size_op_l4);


        //Create the Filter for 3x3 convolution 
        filterL4[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l4, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l4_im2col, im2colL4, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l4_d, filterL4, &err);
        
        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_4*K_D_4;
        int argI_H = K_D_4*K_D_4;
        int argI_W = dG_w*dG_h;
        int argO_W = dG_w*dG_h;
        //printf("Running GEMM Layer2Depth (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_2*K_D_2),(dG_h*dG_w),1,(K_D_2*K_D_2)); 

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
        err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argF_H);
        err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argF_W);
        err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argI_H);
        err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argI_W);
        err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&argO_W);

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
        TotalTime +=end-start;

        //Retrieve result from device
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l4, outputL4_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l1; i++,jf++)

        {    

            outputL4[jf] = outputL4_eachFilter[i];
        }
        
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer done %d\n",itr);
}

//point wise layer
void Layer5( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL4,CHANNELS_5,H_5,W_5,K_D_5,1,0,im2col5);

    printf("Feature Map Dim H_5 %d \t W_5 %d \t K_D_5 %d\n",H_5,W_5,K_D_5);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_5*K_D_5),(dG_h*dG_w));

    filter5[0]=1;

    printf("Running GEMM Layer5Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_5*K_D_5*CHANNELS_5),(dG_h*dG_w),1,(K_D_5*K_D_5*CHANNELS_5)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l5_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l5_im2col, im2col5, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l5_p, filter5, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_5;
    int argF_W = K_D_5*K_D_5*CHANNELS_5;
    int argI_H = K_D_5*K_D_5*CHANNELS_5;
    int argI_W = dG_w*dG_h;
    int argO_W = dG_w*dG_h;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argF_H);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argF_W);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argI_H);
    err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argI_W);
    err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&argO_W);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    //set the local and globar work group size 
    localWorkSize[0] = 2;
    localWorkSize[1] = 2;
    globalWorkSize[0] = dG_w*dG_h;
    globalWorkSize[1] = NO_OF_FILTERS_5;

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);
    clFinish(commands);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    TotalTime +=end-start;
    kernelExecTimeNs = end-start;


    printf("time in  %0.3f nanossec\n", kernelExecTimeNs);
    printf("time in  %0.3f nanossec\n", TotalTime);
    //Retrieve result from device
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l5_out_p, output5, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 5 done %d\n",itr);
}
//Layre 6 depthwise s1
void Layer6( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l6_d = K_D_6 * K_D_6;
    mem_size_filter_l6_d = sizeof(unsigned char) * size_filter_l6_d;
    filterL6 = (unsigned char*)malloc(mem_size_filter_l6_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL6,1,H_6,W_6,K_D_6,1,1,im2colL6);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l6 = dG_h * dG_w;
        mem_size_op_l6 = sizeof(unsigned char) * size_op_l6;
        outputL6_eachFilter = (unsigned char*) malloc(mem_size_op_l6);


        //Create the Filter for 3x3 convolution 
        filterL6[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l6, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l6_im2col, im2colL6, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l6_d, filterL6, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_6*K_D_6;
        int argI_H = K_D_6*K_D_6;
        int argI_W = dG_w*dG_h;
        int argO_W = dG_w*dG_h;
      
        //printf("Running GEMM Layer2Depth (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_2*K_D_2),(dG_h*dG_w),1,(K_D_2*K_D_2)); 

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
        err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argF_H);
        err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argF_W);
        err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argI_H);
        err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argI_W);
        err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&argO_W);

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
        TotalTime +=end-start;

        //Retrieve result from device
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l6, outputL6_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l1; i++,jf++)

        {    

            outputL6[jf] = outputL6_eachFilter[i];
        }
        
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer -6 done %d\n",itr);
}

//point wise layer
void Layer7( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL6,CHANNELS_7,H_7,W_7,K_D_7,1,0,im2col7);

    printf("Feature Map Dim H_7 %d \t W_7 %d \t K_D_7 %d\n",H_7,W_7,K_D_7);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_7*K_D_7),(dG_h*dG_w));

    filter7[0]=1;

    printf("Running GEMM Layer7 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_7*K_D_7*CHANNELS_7),(dG_h*dG_w),1,(K_D_7*K_D_7*CHANNELS_7)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l7_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l7_im2col, im2col7, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l7_p, filter7, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_7;
    int argF_W = K_D_7*K_D_7*CHANNELS_7;
    int argI_H = K_D_7*K_D_7*CHANNELS_7;
    int argI_W = dG_w*dG_h;
    int argO_W = dG_w*dG_h;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argF_H);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argF_W);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argI_H);
    err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argI_W);
    err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&argO_W);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    //set the local and globar work group size 
    localWorkSize[0] = 2;
    localWorkSize[1] = 2;
    globalWorkSize[0] = dG_w*dG_h;
    globalWorkSize[1] = NO_OF_FILTERS_7;

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);
    clFinish(commands);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    TotalTime +=end-start;
    kernelExecTimeNs = end-start;


    printf("time in  %0.3f nanossec\n", kernelExecTimeNs);
    printf("time in  %0.3f nanossec\n", TotalTime);
    //Retrieve result from device
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l7_out_p, output7, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 7 done %d\n",itr);
}

//Layer 8 depthwise s2
void Layer8( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l8_d = K_D_8 * K_D_8;
    mem_size_filter_l8_d = sizeof(unsigned char) * size_filter_l8_d;
    filterL8 = (unsigned char*)malloc(mem_size_filter_l8_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL8,1,H_8,W_8,K_D_8,2,1,im2colL8);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l8 = dG_h * dG_w;
        mem_size_op_l8 = sizeof(unsigned char) * size_op_l8;
        outputL8_eachFilter = (unsigned char*) malloc(mem_size_op_l8);


        //Create the Filter for 3x3 convolution 
        filterL8[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l8, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l8_im2col, im2colL8, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l8_d, filterL8, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_8*K_D_8;
        int argI_H = K_D_8*K_D_8;
        int argI_W = dG_w*dG_h;
        int argO_W = dG_w*dG_h;
        //printf("Running GEMM Layer2Depth (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_2*K_D_2),(dG_h*dG_w),1,(K_D_2*K_D_2)); 

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
        err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argF_H);
        err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argF_W);
        err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argI_H);
        err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argI_W);
        err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&argO_W);

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
        TotalTime +=end-start;

        //Retrieve result from device
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l8, outputL8_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l1; i++,jf++)

        {    

            outputL8[jf] = outputL8_eachFilter[i];
        }
        
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 8 done %d\n",itr);
}

//point wise layer
void Layer9( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL8,CHANNELS_9,H_9,W_9,K_D_9,1,0,im2col9);

    printf("Feature Map Dim H_9 %d \t W_9 %d \t K_D_9 %d\n",H_9,W_9,K_D_9);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_9*K_D_9),(dG_h*dG_w));

    filter9[0]=1;

    printf("Running GEMM Layer9 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_9*K_D_9*CHANNELS_9),(dG_h*dG_w),1,(K_D_9*K_D_9*CHANNELS_9)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l9_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l9_im2col, im2col9, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l9_p, filter9, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_9;
    int argF_W = K_D_9*K_D_9*CHANNELS_9;
    int argI_H = K_D_9*K_D_9*CHANNELS_9;
    int argI_W = dG_w*dG_h;
    int argO_W = dG_w*dG_h;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argF_H);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argF_W);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argI_H);
    err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argI_W);
    err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&argO_W);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    //set the local and globar work group size 
    localWorkSize[0] = 2;
    localWorkSize[1] = 2;
    globalWorkSize[0] = dG_w*dG_h;
    globalWorkSize[1] = NO_OF_FILTERS_9;

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);
    clFinish(commands);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    TotalTime +=end-start;
    kernelExecTimeNs = end-start;


    printf("time in  %0.3f nanossec\n", kernelExecTimeNs);
    printf("time in  %0.3f nanossec\n", TotalTime);
    //Retrieve result from device
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l9_out_p, output9, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 9 done %d\n",itr);
}

//Layer 10 depthwise s1
void Layer10( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l10_d = K_D_10 * K_D_10;
    mem_size_filter_l10_d = sizeof(unsigned char) * size_filter_l10_d;
    filterL10 = (unsigned char*)malloc(mem_size_filter_l10_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL10,1,H_10,W_10,K_D_10,1,1,im2colL10);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l10 = dG_h * dG_w;
        mem_size_op_l10 = sizeof(unsigned char) * size_op_l10;
        outputL10_eachFilter = (unsigned char*) malloc(mem_size_op_l10);


        //Create the Filter for 3x3 convolution 
        filterL10[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l10, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l10_im2col, im2colL10, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l10_d, filterL10, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_10*K_D_10;
        int argI_H = K_D_10*K_D_10;
        int argI_W = dG_w*dG_h;
        int argO_W = dG_w*dG_h;
        //printf("Running GEMM Layer2Depth (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_2*K_D_2),(dG_h*dG_w),1,(K_D_2*K_D_2)); 

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
        err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&argF_H);
        err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&argF_W);
        err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&argI_H);
        err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&argI_W);
        err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&argO_W);

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
        TotalTime +=end-start;

        //Retrieve result from device
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l10, outputL10_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l1; i++,jf++)

        {    

            outputL10[jf] = outputL10_eachFilter[i];
        }
        
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 10 done %d\n",itr);
}

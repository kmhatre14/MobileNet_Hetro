
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
/*Layer11 Pointwise*/
#define W_11 28
#define H_11 28
#define K_D_11 1
#define CHANNELS_11 256
#define NO_OF_FILTERS_11 256 //Channels of input
/*Layer12 Depthwise s2*/
#define W_12 28
#define H_12 28
#define K_D_12 3
#define CHANNELS_12 256
#define NO_OF_FILTERS_12 256 //Channels of input
/*Layer13 Pointwise*/
#define W_13 14
#define H_13 14
#define K_D_13 1
#define CHANNELS_13 256
#define NO_OF_FILTERS_13 512 //Channels of input
/*Layer14 Depthwise*/
#define W_14 14
#define H_14 14
#define K_D_14 3
#define CHANNELS_14 512
#define NO_OF_FILTERS_14 512 //Channels of input
/*Layer15 Pointwise*/
#define W_15 14
#define H_15 14
#define K_D_15 1
#define CHANNELS_15 512
#define NO_OF_FILTERS_15 512 //Channels of input

/*Layer16 Depthwise*/
#define W_16 14
#define H_16 14
#define K_D_16 3
#define CHANNELS_16 512
#define NO_OF_FILTERS_16 512 //Channels of input
/*Layer17 Pointwise*/
#define W_17 14
#define H_17 14
#define K_D_17 1
#define CHANNELS_17 512
#define NO_OF_FILTERS_17 512 //Channels of input

/*Layer18 Depthwise*/
#define W_18 14
#define H_18 14
#define K_D_18 3
#define CHANNELS_18 512
#define NO_OF_FILTERS_18 512 //Channels of input
/*Layer19 Pointwise*/
#define W_19 14
#define H_19 14
#define K_D_19 1
#define CHANNELS_19 512
#define NO_OF_FILTERS_19 512 //Channels of input

/*Layer20 Depthwise*/
#define W_20 14
#define H_20 14
#define K_D_20 3
#define CHANNELS_20 512
#define NO_OF_FILTERS_20 512 //Channels of input
/*Layer21 Pointwise*/
#define W_21 14
#define H_21 14
#define K_D_21 1
#define CHANNELS_21 512
#define NO_OF_FILTERS_21 512 //Channels of input

/*Layer22 Depthwise*/
#define W_22 14
#define H_22 14
#define K_D_22 3
#define CHANNELS_22 512
#define NO_OF_FILTERS_22 512 //Channels of input
/*Layer23 Pointwise*/
#define W_23 14
#define H_23 14
#define K_D_23 1
#define CHANNELS_23 512
#define NO_OF_FILTERS_23 512 //Channels of input
/*Layer24 Depthwise*/
#define W_24 14
#define H_24 14
#define K_D_24 3
#define CHANNELS_24 512
#define NO_OF_FILTERS_24 512
/*Layer25 Pointwise*/
#define W_25 7
#define H_25 7
#define K_D_25 1
#define CHANNELS_25 512
#define NO_OF_FILTERS_25 1024
// Layer26 Depthwise
#define W_26 7
#define H_26 7
#define K_D_26 3
#define CHANNELS_26 1024
#define NO_OF_FILTERS_26 1024
//Layer27 Pointwise 
#define W_27 7
#define H_27 7
#define K_D_27 1
#define CHANNELS_27 1024
#define NO_OF_FILTERS_27 1024
//Layer28 AVG POOL 
#define W_28 7
#define H_28 7
#define K_D_28 1
#define CHANNELS_28 1024
//Layer29 Fully Connected 
#define ELEMENTS 1024
#define CLASSES 1000


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
void Layer11( void );
void Layer12( void );
void Layer13( void );
void Layer14( void );
void Layer15( void );
void Layer16( void );
void Layer17( void );
void Layer18( void );
void Layer19( void );
void Layer20( void );
void Layer21( void );
void Layer22( void );
void Layer23( void );
void Layer24( void );
void Layer25( void );
void Layer26( void );
void Layer27( void );
void Layer28( void );
void Layer29( void );
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
unsigned char* outputL9;

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

//For layer 11 pointwise
unsigned int size_l11_im2col;
unsigned int mem_size_l11_im2col;
unsigned char* im2col11;

unsigned int size_l11_out_p;
unsigned int mem_size_l11_out_p;
unsigned char* output11;

unsigned int size_filter_l11_p;
unsigned int mem_size_filter_l11_p;
unsigned char* filter11;

//For Layer 12 Depthwise
unsigned int size_l12_im2col;
unsigned int mem_size_l12_im2col;
unsigned char* im2colL12;

unsigned int size_l12_out;
unsigned int mem_size_l12_out;
unsigned char* outputL12;

unsigned int size_filter_l12_d;
unsigned int mem_size_filter_l12_d;
unsigned char* filterL12;

unsigned int size_op_l12;
unsigned int mem_size_op_l12;
unsigned char* outputL12_eachFilter;

//For layer 13 pointwise
unsigned int size_l13_im2col;
unsigned int mem_size_l13_im2col;
unsigned char* im2col13;

unsigned int size_l13_out_p;
unsigned int mem_size_l13_out_p;
unsigned char* outputL13;

unsigned int size_filter_l13_p;
unsigned int mem_size_filter_l13_p;
unsigned char* filter13;

//For Layer 14 Depthwise
unsigned int size_l14_im2col;
unsigned int mem_size_l14_im2col;
unsigned char* im2colL14;

unsigned int size_l14_out;
unsigned int mem_size_l14_out;
unsigned char* outputL14;

unsigned int size_filter_l14_d;
unsigned int mem_size_filter_l14_d;
unsigned char* filterL14;

unsigned int size_op_l14;
unsigned int mem_size_op_l14;
unsigned char* outputL14_eachFilter;

//For layer 15 pointwise
unsigned int size_l15_im2col;
unsigned int mem_size_l15_im2col;
unsigned char* im2col15;

unsigned int size_l15_out_p;
unsigned int mem_size_l15_out_p;
unsigned char* outputL15;

unsigned int size_filter_l15_p;
unsigned int mem_size_filter_l15_p;
unsigned char* filter15;

//For Layer 16 Depthwise
unsigned int size_l16_im2col;
unsigned int mem_size_l16_im2col;
unsigned char* im2colL16;

unsigned int size_l16_out;
unsigned int mem_size_l16_out;
unsigned char* outputL16;

unsigned int size_filter_l16_d;
unsigned int mem_size_filter_l16_d;
unsigned char* filterL16;

unsigned int size_op_l16;
unsigned int mem_size_op_l16;
unsigned char* outputL16_eachFilter;

//For layer 17 pointwise
unsigned int size_l17_im2col;
unsigned int mem_size_l17_im2col;
unsigned char* im2col17;

unsigned int size_l17_out_p;
unsigned int mem_size_l17_out_p;
unsigned char* outputL17;

unsigned int size_filter_l17_p;
unsigned int mem_size_filter_l17_p;
unsigned char* filter17;

//For Layer 18 Depthwise
unsigned int size_l18_im2col;
unsigned int mem_size_l18_im2col;
unsigned char* im2colL18;

unsigned int size_l18_out;
unsigned int mem_size_l18_out;
unsigned char* outputL18;

unsigned int size_filter_l18_d;
unsigned int mem_size_filter_l18_d;
unsigned char* filterL18;

unsigned int size_op_l18;
unsigned int mem_size_op_l18;
unsigned char* outputL18_eachFilter;

//For layer 19 pointwise
unsigned int size_l19_im2col;
unsigned int mem_size_l19_im2col;
unsigned char* im2col19;

unsigned int size_l19_out_p;
unsigned int mem_size_l19_out_p;
unsigned char* outputL19;

unsigned int size_filter_l19_p;
unsigned int mem_size_filter_l19_p;
unsigned char* filter19;

//For Layer 20 Depthwise
unsigned int size_l20_im2col;
unsigned int mem_size_l20_im2col;
unsigned char* im2colL20;

unsigned int size_l20_out;
unsigned int mem_size_l20_out;
unsigned char* outputL20;

unsigned int size_filter_l20_d;
unsigned int mem_size_filter_l20_d;
unsigned char* filterL20;

unsigned int size_op_l20;
unsigned int mem_size_op_l20;
unsigned char* outputL20_eachFilter;

//For layer 21 pointwise
unsigned int size_l21_im2col;
unsigned int mem_size_l21_im2col;
unsigned char* im2col21;

unsigned int size_l21_out_p;
unsigned int mem_size_l21_out_p;
unsigned char* outputL21;

unsigned int size_filter_l21_p;
unsigned int mem_size_filter_l21_p;
unsigned char* filter21;

//For Layer 22 Depthwise
unsigned int size_l22_im2col;
unsigned int mem_size_l22_im2col;
unsigned char* im2colL22;

unsigned int size_l22_out;
unsigned int mem_size_l22_out;
unsigned char* outputL22;

unsigned int size_filter_l22_d;
unsigned int mem_size_filter_l22_d;
unsigned char* filterL22;

unsigned int size_op_l22;
unsigned int mem_size_op_l22;
unsigned char* outputL22_eachFilter;

//For layer 23 pointwise
unsigned int size_l23_im2col;
unsigned int mem_size_l23_im2col;
unsigned char* im2col23;

unsigned int size_l23_out_p;
unsigned int mem_size_l23_out_p;
unsigned char* outputL23;

unsigned int size_filter_l23_p;
unsigned int mem_size_filter_l23_p;
unsigned char* filter23;
//For Layer 24 Depthwise
unsigned int size_l24_im2col;
unsigned int mem_size_l24_im2col;
unsigned char* im2colL24;

unsigned int size_l24_out;
unsigned int mem_size_l24_out;
unsigned char* outputL24;

unsigned int size_filter_l24_d;
unsigned int mem_size_filter_l24_d;
unsigned char* filterL24;

unsigned int size_op_l24;
unsigned int mem_size_op_l24;
unsigned char* outputL24_eachFilter;
//For layer 25 pointwise


unsigned int size_l25_im2col;
unsigned int mem_size_l25_im2col;
unsigned char* im2col25;

unsigned int size_l25_out_p;
unsigned int mem_size_l25_out_p;
unsigned char* outputL25;

unsigned int size_filter_l25_p;
unsigned int mem_size_filter_l25_p;
unsigned char* filter25;

//Layer 26 Depthwise
unsigned int size_l26_im2col;
unsigned int mem_size_l26_im2col;
unsigned char* im2colL26;

unsigned int size_l26_out;
unsigned int mem_size_l26_out;
unsigned char* outputL26;

unsigned int size_filter_l26_d;
unsigned int mem_size_filter_l26_d;
unsigned char* filterL26;

unsigned int size_op_l26;
unsigned int mem_size_op_l26;
unsigned char* outputL26_eachFilter;

//Layer 27 Pointwise
unsigned int size_l27_im2col;
unsigned int mem_size_l27_im2col;
unsigned char* im2col27;

unsigned int size_l27_out_p;
unsigned int mem_size_l27_out_p;
unsigned char* outputL27;

unsigned int size_filter_l27_p;
unsigned int mem_size_filter_l27_p;
unsigned char* filter27;

//Layer 28 Avg Pool
unsigned int size_l28_out_p;
unsigned int mem_size_l28_out_p;
unsigned char* outputL28;

//Layer 29 Fully Connected
unsigned int size_l29_out_p;
unsigned int mem_size_l29_out_p;
unsigned char* outputL29;

unsigned int size_filter_l29_p;
unsigned int mem_size_filter_l29_p;
unsigned char* filter29;

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
    outputL9 = (unsigned char*) malloc(mem_size_l9_out_p);      

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
    
    Layer10();

  /*******************Layer 10 Ends*********************/
  
  /*******************Layer 11 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l11_im2col = K_D_11*K_D_11*H_11*W_11*CHANNELS_11;
    mem_size_l11_im2col = sizeof(unsigned char) * size_l11_im2col;
    im2col11 = (unsigned char*) malloc(mem_size_l11_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l11_out_p = H_11*W_11*NO_OF_FILTERS_11;
    mem_size_l11_out_p = sizeof(unsigned char) * size_l11_out_p;
    output11 = (unsigned char*) malloc(mem_size_l11_out_p);      

    //Allocate host memory for filter
    size_filter_l11_p = K_D_11 * K_D_11 * CHANNELS_11 *NO_OF_FILTERS_11 ;
    mem_size_filter_l11_p = sizeof(unsigned char) * size_filter_l11_p;
    filter11 = (unsigned char*)malloc(mem_size_filter_l11_p);

    Layer11();
  /*******************Layer 11 Ends****************/  
  
  /*******************Layer 12 Depthwise s2****************/

    //Allocate host memory op of 1st layer im2col
    size_l12_im2col = K_D_12*K_D_12*H_12*W_12;
    mem_size_l12_im2col = sizeof(unsigned char) * size_l12_im2col;
    im2colL12 = (unsigned char*) malloc(mem_size_l12_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l12_out = H_12*W_12*NO_OF_FILTERS_12;
    mem_size_l12_out = sizeof(unsigned char) * size_l12_out;
    outputL12 = (unsigned char*) malloc(mem_size_l12_out);      
    
    Layer12();

  /*******************Layer 12 Ends*********************/
  
  /*******************Layer 13 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l13_im2col = K_D_13*K_D_13*H_13*W_13*CHANNELS_13;
    mem_size_l13_im2col = sizeof(unsigned char) * size_l13_im2col;
    im2col13 = (unsigned char*) malloc(mem_size_l13_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l13_out_p = H_13*W_13*NO_OF_FILTERS_13;
    mem_size_l13_out_p = sizeof(unsigned char) * size_l13_out_p;
    outputL13 = (unsigned char*) malloc(mem_size_l13_out_p);      

    //Allocate host memory for filter
    size_filter_l13_p = K_D_13 * K_D_13 * CHANNELS_13 *NO_OF_FILTERS_13 ;
    mem_size_filter_l13_p = sizeof(unsigned char) * size_filter_l13_p;
    filter13 = (unsigned char*)malloc(mem_size_filter_l13_p);

    Layer13();
  /*******************Layer 13 Ends****************/

   /*******************Layer 14 Depthwise s2****************/

    //Allocate host memory op of 1st layer im2col
    size_l14_im2col = K_D_14*K_D_14*H_14*W_14;
    mem_size_l14_im2col = sizeof(unsigned char) * size_l14_im2col;
    im2colL14 = (unsigned char*) malloc(mem_size_l14_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l14_out = H_14*W_14*NO_OF_FILTERS_14;
    mem_size_l14_out = sizeof(unsigned char) * size_l14_out;
    outputL14 = (unsigned char*) malloc(mem_size_l14_out);      
    
    Layer14();

   /*******************Layer 14 Ends*********************/
   
   /*******************Layer 15 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l15_im2col = K_D_15*K_D_15*H_15*W_15*CHANNELS_15;
    mem_size_l15_im2col = sizeof(unsigned char) * size_l15_im2col;
    im2col15 = (unsigned char*) malloc(mem_size_l15_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l15_out_p = H_15*W_15*NO_OF_FILTERS_15;
    mem_size_l15_out_p = sizeof(unsigned char) * size_l15_out_p;
    outputL15 = (unsigned char*) malloc(mem_size_l15_out_p);      

    //Allocate host memory for filter
    size_filter_l15_p = K_D_15 * K_D_15 * CHANNELS_15 *NO_OF_FILTERS_15 ;
    mem_size_filter_l15_p = sizeof(unsigned char) * size_filter_l15_p;
    filter15 = (unsigned char*)malloc(mem_size_filter_l15_p);

    Layer15();
   /*******************Layer 15 Ends****************/

   /*******************Layer 16 Depthwise s2****************/

    //Allocate host memory op of 1st layer im2col
    size_l16_im2col = K_D_16*K_D_16*H_16*W_16;
    mem_size_l16_im2col = sizeof(unsigned char) * size_l16_im2col;
    im2colL16 = (unsigned char*) malloc(mem_size_l16_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l16_out = H_16*W_16*NO_OF_FILTERS_16;
    mem_size_l16_out = sizeof(unsigned char) * size_l16_out;
    outputL16 = (unsigned char*) malloc(mem_size_l16_out);      
    
    Layer16();

   /*******************Layer 16 Ends*********************/

   /*******************Layer 17 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l17_im2col = K_D_17*K_D_17*H_17*W_17*CHANNELS_17;
    mem_size_l17_im2col = sizeof(unsigned char) * size_l17_im2col;
    im2col17 = (unsigned char*) malloc(mem_size_l17_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l17_out_p = H_17*W_17*NO_OF_FILTERS_17;
    mem_size_l17_out_p = sizeof(unsigned char) * size_l17_out_p;
    outputL17 = (unsigned char*) malloc(mem_size_l17_out_p);      

    //Allocate host memory for filter
    size_filter_l17_p = K_D_17 * K_D_17 * CHANNELS_17 *NO_OF_FILTERS_17 ;
    mem_size_filter_l17_p = sizeof(unsigned char) * size_filter_l17_p;
    filter17 = (unsigned char*)malloc(mem_size_filter_l17_p);

    Layer17();
   /*******************Layer 17 Ends****************/

   
   /*******************Layer 18 Depthwise s2****************/

    //Allocate host memory op of 1st layer im2col
    size_l18_im2col = K_D_18*K_D_18*H_18*W_18;
    mem_size_l18_im2col = sizeof(unsigned char) * size_l18_im2col;
    im2colL18 = (unsigned char*) malloc(mem_size_l18_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l18_out = H_18*W_18*NO_OF_FILTERS_18;
    mem_size_l18_out = sizeof(unsigned char) * size_l18_out;
    outputL18 = (unsigned char*) malloc(mem_size_l18_out);      
    
    Layer18();

   /*******************Layer 18 Ends*********************/
   
   /*******************Layer 19 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l19_im2col = K_D_19*K_D_19*H_19*W_19*CHANNELS_19;
    mem_size_l19_im2col = sizeof(unsigned char) * size_l19_im2col;
    im2col19 = (unsigned char*) malloc(mem_size_l19_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l19_out_p = H_19*W_19*NO_OF_FILTERS_19;
    mem_size_l19_out_p = sizeof(unsigned char) * size_l19_out_p;
    outputL19 = (unsigned char*) malloc(mem_size_l19_out_p);      

    //Allocate host memory for filter
    size_filter_l19_p = K_D_19 * K_D_19 * CHANNELS_19 *NO_OF_FILTERS_19 ;
    mem_size_filter_l19_p = sizeof(unsigned char) * size_filter_l19_p;
    filter19 = (unsigned char*)malloc(mem_size_filter_l19_p);

    Layer19();
   /*******************Layer 19 Ends****************/

   /*******************Layer 20 Depthwise s2****************/

    //Allocate host memory op of 1st layer im2col
    size_l20_im2col = K_D_20*K_D_20*H_20*W_20;
    mem_size_l20_im2col = sizeof(unsigned char) * size_l20_im2col;
    im2colL20 = (unsigned char*) malloc(mem_size_l20_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l20_out = H_20*W_20*NO_OF_FILTERS_20;
    mem_size_l20_out = sizeof(unsigned char) * size_l20_out;
    outputL20 = (unsigned char*) malloc(mem_size_l20_out);      
    
    Layer20();

   /*******************Layer 20 Ends*********************/

   /*******************Layer 21 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l21_im2col = K_D_21*K_D_21*H_21*W_21*CHANNELS_21;
    mem_size_l21_im2col = sizeof(unsigned char) * size_l21_im2col;
    im2col21 = (unsigned char*) malloc(mem_size_l21_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l21_out_p = H_21*W_21*NO_OF_FILTERS_21;
    mem_size_l21_out_p = sizeof(unsigned char) * size_l21_out_p;
    outputL21 = (unsigned char*) malloc(mem_size_l21_out_p);      

    //Allocate host memory for filter
    size_filter_l21_p = K_D_21 * K_D_21 * CHANNELS_21 *NO_OF_FILTERS_21 ;
    mem_size_filter_l21_p = sizeof(unsigned char) * size_filter_l21_p;
    filter21 = (unsigned char*)malloc(mem_size_filter_l21_p);

    Layer21();
   /*******************Layer 21 Ends****************/

   
   /*******************Layer 22 Depthwise s2****************/

    //Allocate host memory op of 1st layer im2col
    size_l22_im2col = K_D_22*K_D_22*H_22*W_22;
    mem_size_l22_im2col = sizeof(unsigned char) * size_l22_im2col;
    im2colL22 = (unsigned char*) malloc(mem_size_l22_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l22_out = H_22*W_22*NO_OF_FILTERS_22;
    mem_size_l22_out = sizeof(unsigned char) * size_l22_out;
    outputL22 = (unsigned char*) malloc(mem_size_l22_out);      
    
    Layer22();

   /*******************Layer 22 Ends*********************/

   /*******************Layer 23 pointwise****************/

    //Allocate host memory op of 2st layer im2col
    size_l23_im2col = K_D_23*K_D_23*H_23*W_23*CHANNELS_23;
    mem_size_l23_im2col = sizeof(unsigned char) * size_l23_im2col;
    im2col23 = (unsigned char*) malloc(mem_size_l23_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l23_out_p = H_23*W_23*NO_OF_FILTERS_23;
    mem_size_l23_out_p = sizeof(unsigned char) * size_l23_out_p;
    outputL23 = (unsigned char*) malloc(mem_size_l23_out_p);      

    //Allocate host memory for filter
    size_filter_l23_p = K_D_23 * K_D_23 * CHANNELS_23 *NO_OF_FILTERS_23 ;
    mem_size_filter_l23_p = sizeof(unsigned char) * size_filter_l23_p;
    filter23 = (unsigned char*)malloc(mem_size_filter_l23_p);

    Layer23();
   /*******************Layer 23 Ends****************/

   /*******************Layer 24 Depthwise s1****************/
    size_l24_im2col = K_D_24*K_D_24*H_24*W_24;
    mem_size_l24_im2col = sizeof(unsigned char) * size_l24_im2col;
    im2colL24 = (unsigned char*) malloc(mem_size_l24_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l24_out = H_24*W_24*NO_OF_FILTERS_24;
    mem_size_l24_out = sizeof(unsigned char) * size_l24_out;
    outputL24 = (unsigned char*) malloc(mem_size_l24_out);  

    Layer24();
   /*******************Layer 24 Depthwise ENDS s1****************/

   /*******************Layer 25 Pointwise s1****************/


    size_l25_im2col = K_D_25*K_D_25*H_25*W_25*CHANNELS_25;
    mem_size_l25_im2col = sizeof(unsigned char) * size_l25_im2col;
    im2col25 = (unsigned char*) malloc(mem_size_l25_im2col);  
  
    //Allocate host memory op of Pointwise layer
    size_l25_out_p = H_25*W_25*NO_OF_FILTERS_25;
    mem_size_l25_out_p = sizeof(unsigned char) * size_l25_out_p;
    outputL25 = (unsigned char*) malloc(mem_size_l25_out_p);      

    //Allocate host memory for filter
    size_filter_l25_p = K_D_25 * K_D_25 * CHANNELS_25 *NO_OF_FILTERS_25 ;
    mem_size_filter_l25_p = sizeof(unsigned char) * size_filter_l25_p;
    filter25 = (unsigned char*)malloc(mem_size_filter_l25_p);

    Layer25();

   /*******************Layer 25 Pointwise ENDS****************/
      /*******************Layer 26 depthwise****************/
    size_l26_im2col = K_D_26*K_D_26*H_26*W_26;
    mem_size_l26_im2col = sizeof(unsigned char) * size_l26_im2col;
    im2colL26 = (unsigned char*) malloc(mem_size_l26_im2col);      
    
    //Allocate host memory op of Depthewise2 layer
    size_l26_out = H_26*W_26*NO_OF_FILTERS_26;
    mem_size_l26_out = sizeof(unsigned char) * size_l26_out;
    outputL26 = (unsigned char*) malloc(mem_size_l26_out);      
    
    Layer26();
    /*******************Layer 26 depthwise ENDS****************/

    /*******************Layer 27 Pointwise ****************/

    size_l27_im2col = K_D_27*K_D_27*H_27*W_27*CHANNELS_27;
    mem_size_l27_im2col = sizeof(unsigned char) * size_l27_im2col;
    im2col27 = (unsigned char*) malloc(mem_size_l27_im2col);  
  
    //Allocate host memory op of Pointwise layer

    size_l27_out_p = H_27*W_27*NO_OF_FILTERS_27;
    mem_size_l27_out_p = sizeof(unsigned char) * size_l27_out_p;
    outputL27 = (unsigned char*) malloc(mem_size_l27_out_p);      

    //Allocate host memory for filter
    size_filter_l27_p = K_D_27 * K_D_27 * CHANNELS_27 *NO_OF_FILTERS_27 ;
    mem_size_filter_l27_p = sizeof(unsigned char) * size_filter_l27_p;
    filter27 = (unsigned char*)malloc(mem_size_filter_l27_p);

    Layer27();

    /*******************Layer 27 pointwise ENDS****************/

    /*******************Layer 28 Avg Pool ****************/

    size_l28_out_p = CHANNELS_28;
    mem_size_l28_out_p = sizeof(unsigned char) * size_l28_out_p;
    outputL28 = (unsigned char*) malloc(mem_size_l28_out_p);      

    Layer28();

    /*******************Layer 28 AVg pool ENDS****************/


    /*******************Layer 29 Fully connected ****************/

    //Allocate host memory for filter
    size_filter_l29_p = CLASSES*ELEMENTS;
    mem_size_filter_l29_p = sizeof(unsigned char) * size_filter_l29_p;
    filter29 = (unsigned char*)malloc(mem_size_filter_l29_p);

    size_l29_out_p = CLASSES;
    mem_size_l29_out_p = sizeof(unsigned char) * size_l29_out_p;
    outputL29 = (unsigned char*) malloc(mem_size_l29_out_p);      

    Layer29();

    /*******************Layer 29 Fully Connected ENDS****************/


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
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l9_out_p, outputL9, 0, NULL, NULL);
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
        im2col_cpu(outputL9,1,H_10,W_10,K_D_10,1,1,im2colL10);

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

        for(i=0; i<size_op_l10; i++,jf++)

        {    

            outputL10[jf] = outputL10_eachFilter[i];
        }
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 10 done %d\n",itr);
}

//point wise layer
void Layer11( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL10,CHANNELS_11,H_11,W_11,K_D_11,1,0,im2col11);

    printf("Feature Map Dim H_11 %d \t W_11 %d \t K_D_11 %d\n",H_11,W_11,K_D_11);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_11*K_D_11),(dG_h*dG_w));

    filter11[0]=1;

    printf("Running GEMM Layer 11 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_11*K_D_11*CHANNELS_11),(dG_h*dG_w),1,(K_D_11*K_D_11*CHANNELS_11)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l11_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l11_im2col, im2col11, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l11_p, filter11, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_11;
    int argF_W = K_D_11*K_D_11*CHANNELS_11;
    int argI_H = K_D_11*K_D_11*CHANNELS_11;
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
    globalWorkSize[1] = NO_OF_FILTERS_11;

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
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l11_out_p, output11, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 11 done %d\n",itr);
}

//Layer 12 depthwise s2
void Layer12( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l12_d = K_D_12 * K_D_12;
    mem_size_filter_l12_d = sizeof(unsigned char) * size_filter_l12_d;
    filterL12 = (unsigned char*)malloc(mem_size_filter_l12_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL12,1,H_12,W_12,K_D_12,2,1,im2colL12);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l12 = dG_h * dG_w;
        mem_size_op_l12 = sizeof(unsigned char) * size_op_l12;
        outputL12_eachFilter = (unsigned char*) malloc(mem_size_op_l12);


        //Create the Filter for 3x3 convolution 
        filterL12[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l12, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l12_im2col, im2colL12, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l12_d, filterL12, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_12*K_D_12;
        int argI_H = K_D_12*K_D_12;
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
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l12, outputL12_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l12; i++,jf++)

        {    

            outputL12[jf] = outputL12_eachFilter[i];
        }
        
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 12 done %d\n",itr);
}

//point wise layer
void Layer13( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL12,CHANNELS_13,H_13,W_13,K_D_13,1,0,im2col13);

    printf("Feature Map Dim H_13 %d \t W_13 %d \t K_D_13 %d\n",H_13,W_13,K_D_13);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_13*K_D_13),(dG_h*dG_w));

    filter13[0]=1;

    printf("Running GEMM Layer 13 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_13*K_D_13*CHANNELS_13),(dG_h*dG_w),1,(K_D_13*K_D_13*CHANNELS_13)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l13_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l13_im2col, im2col13, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l13_p, filter13, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_13;
    int argF_W = K_D_13*K_D_13*CHANNELS_13;
    int argI_H = K_D_13*K_D_13*CHANNELS_13;
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
    globalWorkSize[1] = NO_OF_FILTERS_13;

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
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l13_out_p, outputL13, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 13 done %d\n",itr);
}
//Layer 14 depthwise 
void Layer14( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l14_d = K_D_14 * K_D_14;
    mem_size_filter_l14_d = sizeof(unsigned char) * size_filter_l14_d;
    filterL14 = (unsigned char*)malloc(mem_size_filter_l14_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL13,1,H_14,W_14,K_D_14,1,1,im2colL14);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l14 = dG_h * dG_w;
        mem_size_op_l14 = sizeof(unsigned char) * size_op_l14;
        outputL14_eachFilter = (unsigned char*) malloc(mem_size_op_l14);


        //Create the Filter for 3x3 convolution 
        filterL14[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l14, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l14_im2col, im2colL14, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l14_d, filterL14, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_14*K_D_14;
        int argI_H = K_D_14*K_D_14;
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
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l14, outputL14_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l14; i++,jf++)

        {    

            outputL14[jf] = outputL14_eachFilter[i];
        }
        
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 14 done %d\n",itr);
}

//point wise layer
void Layer15( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL14,CHANNELS_15,H_15,W_15,K_D_15,1,0,im2col15);

    printf("Feature Map Dim H_15 %d \t W_15 %d \t K_D_15 %d\n",H_15,W_15,K_D_15);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_15*K_D_15),(dG_h*dG_w));

    filter15[0]=1;

    printf("Running GEMM Layer15 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_15*K_D_15*CHANNELS_15),(dG_h*dG_w),1,(K_D_15*K_D_15*CHANNELS_15)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l15_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l15_im2col, im2col15, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l15_p, filter15, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_15;
    int argF_W = K_D_15*K_D_15*CHANNELS_15;
    int argI_H = K_D_15*K_D_15*CHANNELS_15;
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
    globalWorkSize[1] = NO_OF_FILTERS_15;

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
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l15_out_p, outputL15, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 15 done %d\n",itr);
}
//Layer 16 depthwise 
void Layer16( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l16_d = K_D_16 * K_D_16;
    mem_size_filter_l16_d = sizeof(unsigned char) * size_filter_l16_d;
    filterL16 = (unsigned char*)malloc(mem_size_filter_l16_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL15,1,H_16,W_16,K_D_16,1,1,im2colL16);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l16 = dG_h * dG_w;
        mem_size_op_l16 = sizeof(unsigned char) * size_op_l16;
        outputL16_eachFilter = (unsigned char*) malloc(mem_size_op_l16);


        //Create the Filter for 3x3 convolution 
        filterL16[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l16, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l16_im2col, im2colL16, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l16_d, filterL16, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_16*K_D_16;
        int argI_H = K_D_16*K_D_16;
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
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l16, outputL16_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l16; i++,jf++)

        {    

            outputL16[jf] = outputL16_eachFilter[i];
        }
        
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 16 done %d\n",itr);
}

//point wise layer
void Layer17( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL16,CHANNELS_17,H_17,W_17,K_D_17,1,0,im2col17);

    printf("Feature Map Dim H_17 %d \t W_17 %d \t K_D_17 %d\n",H_17,W_17,K_D_17);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_17*K_D_17),(dG_h*dG_w));

    filter17[0]=1;

    printf("Running GEMM Layer17 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_17*K_D_17*CHANNELS_17),(dG_h*dG_w),1,(K_D_17*K_D_17*CHANNELS_17)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l17_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l17_im2col, im2col17, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l17_p, filter17, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_17;
    int argF_W = K_D_17*K_D_17*CHANNELS_17;
    int argI_H = K_D_17*K_D_17*CHANNELS_17;
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
    globalWorkSize[1] = NO_OF_FILTERS_17;

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
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l17_out_p, outputL17, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 17 done %d\n",itr);
}

//Layer 18 depthwise 
void Layer18( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l18_d = K_D_18 * K_D_18;
    mem_size_filter_l18_d = sizeof(unsigned char) * size_filter_l18_d;
    filterL18 = (unsigned char*)malloc(mem_size_filter_l18_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL17,1,H_18,W_18,K_D_18,1,1,im2colL18);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l18 = dG_h * dG_w;
        mem_size_op_l18 = sizeof(unsigned char) * size_op_l18;
        outputL18_eachFilter = (unsigned char*) malloc(mem_size_op_l18);


        //Create the Filter for 3x3 convolution 
        filterL18[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l18, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l18_im2col, im2colL18, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l18_d, filterL18, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_18*K_D_18;
        int argI_H = K_D_18*K_D_18;
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
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l18, outputL18_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l18; i++,jf++)

        {    

            outputL18[jf] = outputL18_eachFilter[i];
        }
        
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 18 done %d\n",itr);
}

//point wise layer
void Layer19( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL18,CHANNELS_19,H_19,W_19,K_D_19,1,0,im2col19);

    printf("Feature Map Dim H_19 %d \t W_19 %d \t K_D_19 %d\n",H_19,W_19,K_D_19);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_19*K_D_19),(dG_h*dG_w));

    filter19[0]=1;

    printf("Running GEMM Layer19 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_19*K_D_19*CHANNELS_19),(dG_h*dG_w),1,(K_D_19*K_D_19*CHANNELS_19)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l19_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l19_im2col, im2col19, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l19_p, filter19, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_19;
    int argF_W = K_D_19*K_D_19*CHANNELS_19;
    int argI_H = K_D_19*K_D_19*CHANNELS_19;
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
    globalWorkSize[1] = NO_OF_FILTERS_19;

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
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l19_out_p, outputL19, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 19 done %d\n",itr);
}

//Layer 20 depthwise 
void Layer20( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l20_d = K_D_20 * K_D_20;
    mem_size_filter_l20_d = sizeof(unsigned char) * size_filter_l20_d;
    filterL20 = (unsigned char*)malloc(mem_size_filter_l20_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL19,1,H_20,W_20,K_D_20,1,1,im2colL20);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l20 = dG_h * dG_w;
        mem_size_op_l20 = sizeof(unsigned char) * size_op_l20;
        outputL20_eachFilter = (unsigned char*) malloc(mem_size_op_l20);


        //Create the Filter for 3x3 convolution 
        filterL20[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l20, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l20_im2col, im2colL20, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l20_d, filterL20, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_20*K_D_20;
        int argI_H = K_D_20*K_D_20;
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
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l20, outputL20_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l20; i++,jf++)

        {    

            outputL20[jf] = outputL20_eachFilter[i];
        }
        
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 20 done %d\n",itr);
}


//point wise layer
void Layer21( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL20,CHANNELS_21,H_21,W_21,K_D_21,1,0,im2col21);

    printf("Feature Map Dim H_21 %d \t W_21 %d \t K_D_21 %d\n",H_21,W_21,K_D_21);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_21*K_D_21),(dG_h*dG_w));

    filter21[0]=1;

    printf("Running GEMM Layer21 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_21*K_D_21*CHANNELS_21),(dG_h*dG_w),1,(K_D_21*K_D_21*CHANNELS_21)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l21_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l21_im2col, im2col21, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l21_p, filter21, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_21;
    int argF_W = K_D_21*K_D_21*CHANNELS_21;
    int argI_H = K_D_21*K_D_21*CHANNELS_21;
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
    globalWorkSize[1] = NO_OF_FILTERS_21;

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
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l21_out_p, outputL21, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 21 done %d\n",itr);
}

//Layer 22 depthwise 
void Layer22( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l22_d = K_D_22 * K_D_22;
    mem_size_filter_l22_d = sizeof(unsigned char) * size_filter_l22_d;
    filterL22 = (unsigned char*)malloc(mem_size_filter_l22_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL21,1,H_22,W_22,K_D_22,1,1,im2colL22);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l22 = dG_h * dG_w;
        mem_size_op_l22 = sizeof(unsigned char) * size_op_l22;
        outputL22_eachFilter = (unsigned char*) malloc(mem_size_op_l22);


        //Create the Filter for 3x3 convolution 
        filterL22[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l22, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l22_im2col, im2colL22, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l22_d, filterL22, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_22*K_D_22;
        int argI_H = K_D_22*K_D_22;
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
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l22, outputL22_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l22; i++,jf++)

        {    

            outputL22[jf] = outputL22_eachFilter[i];
        }
        
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 22 done %d\n",itr);
}

//point wise layer
void Layer23( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL22,CHANNELS_23,H_23,W_23,K_D_23,1,0,im2col23);

    printf("Feature Map Dim H_23 %d \t W_23 %d \t K_D_23 %d\n",H_23,W_23,K_D_23);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_23*K_D_23),(dG_h*dG_w));

    filter23[0]=1;

    printf("Running GEMM Layer23 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_23*K_D_23*CHANNELS_23),(dG_h*dG_w),1,(K_D_23*K_D_23*CHANNELS_23)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l23_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l23_im2col, im2col23, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l23_p, filter23, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_23;
    int argF_W = K_D_23*K_D_23*CHANNELS_23;
    int argI_H = K_D_23*K_D_23*CHANNELS_23;
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
    globalWorkSize[1] = NO_OF_FILTERS_23;

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
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l23_out_p, outputL23, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 23 done %d\n",itr);
}


//Layer 24 Depthwise s2
void Layer24( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l24_d = K_D_24 * K_D_24;
    mem_size_filter_l24_d = sizeof(unsigned char) * size_filter_l24_d;
    filterL24 = (unsigned char*)malloc(mem_size_filter_l24_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL23,3,H_24,W_24,K_D_24,2,1,im2colL24);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l24 = dG_h * dG_w;
        mem_size_op_l24 = sizeof(unsigned char) * size_op_l24;
        outputL24_eachFilter = (unsigned char*) malloc(mem_size_op_l24);


        //Create the Filter for 3x3 convolution 
        filterL24[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l24, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l24_im2col, im2colL24, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l24_d, filterL24, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_24*K_D_24;
        int argI_H = K_D_24*K_D_24;
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
        printf("%d  %d\n", dG_w,dG_h);
        //set the local and globar work group size 
        localWorkSize[0] = 1;
        localWorkSize[1] = 1;
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
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l24, outputL24_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l24; i++,jf++)

        {    

            outputL24[jf] = outputL24_eachFilter[i];
        }
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 24 done %d\n",itr);

}

    //Layer 25 Pointwise

    void Layer25( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL24,CHANNELS_25,H_25,W_25,K_D_25,1,0,im2col25);

    printf("Feature Map Dim H_25 %d \t W_25 %d \t K_D_25 %d\n",H_25,W_25,K_D_25);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_25*K_D_25),(dG_h*dG_w));

    filter25[0]=1;

    printf("Running GEMM Layer25 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_25*K_D_25*CHANNELS_25),(dG_h*dG_w),1,(K_D_25*K_D_25*CHANNELS_25)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l25_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l25_im2col, im2col25, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l25_p, filter25, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_25;
    int argF_W = K_D_25*K_D_25*CHANNELS_25;
    int argI_H = K_D_25*K_D_25*CHANNELS_25;
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
    localWorkSize[0] = 1;
    localWorkSize[1] = 1;
    globalWorkSize[0] = dG_w*dG_h;
    globalWorkSize[1] = NO_OF_FILTERS_25;

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
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l25_out_p, outputL25, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 25 done %d\n",itr);
}


//Layer 26
 void Layer26( void )
{
    int i,j,jf,itr;

    //Allocate host memory for filter
    size_filter_l26_d = K_D_26 * K_D_26;
    mem_size_filter_l26_d = sizeof(unsigned char) * size_filter_l26_d;
    filterL26 = (unsigned char*)malloc(mem_size_filter_l26_d);
    kernelExecTimeNs = 0;
    //get input for previous layer 
    for(itr=0; itr<32; itr++)
    {
        //Convert each chhannel from input to im2col
        im2col_cpu(outputL25,1,H_26,W_26,K_D_26,2,1,im2colL26);

        //printf("Feature Map Dim H_2 %d \t W_2 %d \t K_D_2 %d\n",H_2,W_2,K_D_2);
        //printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
        //printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_2*K_D_2),(dG_h*dG_w));

        //Allocate host memory for the result C
        size_op_l26 = dG_h * dG_w;
        mem_size_op_l26 = sizeof(unsigned char) * size_op_l26;
        outputL26_eachFilter = (unsigned char*) malloc(mem_size_op_l26);


        //Create the Filter for 3x3 convolution 
        filterL26[0]=1;

        //Call the kernel with the following matrix 
        //Create the input and output arrays in device memory for our calculation
        d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_op_l26, NULL, &err);
        d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l26_im2col, im2colL26, &err);
        d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l26_d, filterL26, &err);

        if (!d_image || !d_filter || !d_output)

        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }    
        //Launch OpenCL kernel
        int argF_H = 1;
        int argF_W = K_D_26*K_D_26;
        int argI_H = K_D_26*K_D_26;
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
        localWorkSize[0] = 1;
        localWorkSize[1] = 1;
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
        err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_op_l26, outputL26_eachFilter, 0, NULL, NULL);
        clFinish(commands);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //Store the data from each convolution to an array

        for(i=0; i<size_op_l26; i++,jf++)

        {    

            outputL26[jf] = outputL26_eachFilter[i];
        }
    }
    printf("time in %0.3f nanossec \n", kernelExecTimeNs);
    printf("Depthwise Layer - 26 done %d\n",itr);

}


//Layer 27 Pointwise
void Layer27( void )
{
    int i,j,jf=0,itr;

    //Convert each chhannel from input to im2col
    im2col_cpu(outputL26,CHANNELS_27,H_27,W_27,K_D_27,1,0,im2col27);

    printf("Feature Map Dim H_27 %d \t W_27 %d \t K_D_27 %d\n",H_27,W_27,K_D_27);
    printf("im2Col FeatureMap Dim H %d \t W %d \n",dG_h,dG_w);
    printf("im2Col Matrix Dim H %d \t W %d \n",(K_D_27*K_D_27),(dG_h*dG_w));

    filter27[0]=1;

    printf("Running GEMM Layer27 Point (%dx%d) and Filter_Matrix (%dx%d) ...\n",(K_D_27*K_D_27*CHANNELS_27),(dG_h*dG_w),1,(K_D_27*K_D_27*CHANNELS_27)); 

    //Call the kernel with the following matrix 
    //Create the input and output arrays in device memory for our calculation
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_l27_out_p, NULL, &err);
    d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_l27_im2col, im2col27, &err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter_l27_p, filter27, &err);

    if (!d_image || !d_filter || !d_output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    //Launch OpenCL kernel
    int argF_H = NO_OF_FILTERS_27;
    int argF_W = K_D_27*K_D_27*CHANNELS_27;
    int argI_H = K_D_27*K_D_27*CHANNELS_27;
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
    localWorkSize[0] = 1;
    localWorkSize[1] = 1;
    globalWorkSize[0] = dG_w*dG_h;
    globalWorkSize[1] = NO_OF_FILTERS_27;

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
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_l27_out_p, outputL27, 0, NULL, NULL);
    clFinish(commands);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Matrix multiplication completed...\n"); 

    printf("PointWise Layer 27 done %d\n",itr);
    printf("Final op size %d\n",size_l27_out_p);
    /*for(i=0;i<7;i++)
    {
        for(j=0;j<7;j++)
        { 
            printf("%d \t" , outputL27[i*dG_w + j]);
        }
        printf("\n");
    }*/

}

//Layer 28 Average Pool 
void Layer28( void )
{
    int i,j,jf=0,itr;
    int avgx;
    for(itr=0;itr<CHANNELS_28;itr++)
    {
        for(i=0;i<W_28;i++)
        {
            for(j=0;j<H_28;j++)
            { 
                avgx += outputL27[i*dG_w + j];
                //printf("%d \t" , outputL27[i*dG_w + j]);

            }
            //printf("\n");
        }
        outputL28[itr] = avgx/(H_28*W_28);
        avgx = 0;
    }
    printf("Layer 28 Done\n");
}


//Layer 29 Fully Connected
void Layer29( void )
{   
    int i,j,jf=0,itr;
    int fcVar;
    filter29[0]=1;
    for(i=0;i<CLASSES;i++)
    {
        for(j=0;j<ELEMENTS;j++)
        {
            outputL29[i]+=outputL28[j]*filter29[j];
        }
    }
    printf("Layer 29 Fully Connected Done\n");
}

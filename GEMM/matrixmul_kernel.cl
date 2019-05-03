
/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
matrixMul(__global unsigned char* ouput, 
          __global unsigned char* img_im2col, 
          __global unsigned char* filter, 
          int argF_H, int argF_W, int argI_H, int argI_W, int argO_W)
{
    int tx = get_global_id(0);//112*112
    int ty = get_global_id(1);//32
    int value = 0;
    for (int k = 0; k < argF_W; ++k)
    {
        int elementA = filter[ty*argF_W + k];
        int elementB = img_im2col[k*argI_W+tx];
        value += elementA * elementB;
    }
    //Considering 128 as a threshould for 0
    value = value/argF_W;   
    //ReLU
    if(value < 128)
    {
        value = 0;
    }
    ouput[ty*argO_W+tx] = value;
}


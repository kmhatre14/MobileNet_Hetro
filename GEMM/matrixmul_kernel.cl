
/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
matrixMul(__global unsigned char* C, 
          __global unsigned char* img_im2col, 
          __global unsigned char* filter, 
          int K_D, int H, int W, int channel)
{
    int tx = get_global_id(0);//158 
    int ty = get_global_id(1);//118
    int value = 0;

    for (int k = 0; k < (K_D*K_D*channel); ++k)
    {
        int elementA = filter[k];
        int elementB = img_im2col[k*H*W + (ty*W+tx)];
        value += elementA * elementB;
        //   if(tx == 1 && ty ==0){
        //       printf("%d %d \n",elementA,elementB);
        //   }
    }
    //write back the answer
    C[ty*W+tx] = value/(K_D*K_D*channel);
}


/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
matrixMul(__global unsigned char* C, 
          __global unsigned char* img_im2col, 
          __global unsigned char* filter, 
          int K_D, int wH, int wW, int kH)
{
  
   int tx = get_global_id(0);//96 
   int ty = get_global_id(1);//13509
 
   // value stores the element that is 
   // computed by the thread
   int value = 0;
   for (int k = 0; k < (K_D*K_D); ++k)
   {
      int elementA = filter[(tx*kH)+k];
      int elementB = img_im2col[k*(wH-K_D+1)*(wW-K_D+1)*3 + ty];
      value += elementA * elementB;
   //   if(tx == 1 && ty ==0){
   //       printf("%d %d \n",elementA,elementB);
   //   }
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[ty+ (tx*kH)] = value;
}

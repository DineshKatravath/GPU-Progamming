#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// I have used cuda_runtime.h instead of cuda/cuda_runtime.h

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void convolute(long int *matrix, long int *filter,long int *result, int h, int w, int c, int r, int s, int k)
{
  // dynamic shared memory used to store filter   
  extern __shared__ long int sharedFilter[];

  int col = threadIdx.x;
  int row = blockIdx.y;

  int filterStartIdx = blockIdx.x * (r*s*c); // starting idx of filter for every filter

  int totalCount = ceil((double)s/w),idx;

  // copying filter matrix into shared memory using all threads in a block, using coalesced memory access of filter
  for(int chanel=0; chanel<c;chanel++){
    for(int i=0;i<r;i++){
      for(int j=0;j<totalCount &&((j*w)+col)<s;j++){
        idx = chanel*r*s + i*s+((j*w)+col);
        sharedFilter[idx] = filter[filterStartIdx+idx];
      }
    }
  }

  // making sure that whole filter is copied into shared memory before accessing
  __syncthreads();

  // idx to store result
  int newIdx = blockIdx.x*(h*w) + blockIdx.y*(w) + threadIdx.x;
  long int sum=0;

  int startX = row - r/2;
  int endX = row + r/2;
  int startY = col - s/2;
  int endY = col + s/2;

  // performing convolution
  for(int chanel=0;chanel<c;chanel++){
    int matrixIdx = chanel*(h*w),filterStart = chanel*r*s;
    matrixIdx += blockIdx.y*(w);
    matrixIdx += threadIdx.x;

    int matrixStartIdx = matrixIdx - s/2;
    matrixStartIdx -= w*(r/2);

    for(int i=startX, x=0 ; i<=endX ;i++,x++){
      int id = matrixStartIdx + x*w;
      for(int j=startY,y=0; i>=0 && i<h && j<=endY; j++,y++,id++){
        if(j>=0 && j<w){
          sum += sharedFilter[filterStart+x*s+y]*matrix[id];
        }
      }
    }
  }

  // storing in result final value.
  result[newIdx]=sum;

}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /****************************************************Start Here***********************************************************/

    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    long int* d_matrix,*d_filter,*d_ans;
    dim3 grid(k,h,1);

    // Memory Allocation on GPU
    cudaMalloc(&d_matrix,sizeof(long int)*h*w*c);
    cudaMalloc(&d_filter,sizeof(long int)*r*s*c*k);
    cudaMalloc(&d_ans,sizeof(long int)*h*w*k);

    // copying Input and Filter Matrices to GPU
    cudaMemcpy(d_matrix,h_mat,h*w*c*sizeof(long int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter,h_filter,r*s*c*k*sizeof(long int),cudaMemcpyHostToDevice);

    // Kernel Launch
    convolute<<<grid,w,r*s*c*sizeof(long int)>>>(d_matrix,d_filter,d_ans,h,w,c,r,s,k);

    // storing ans in CPU
    cudaMemcpy(h_ans,d_ans,h*w*k*sizeof(long int),cudaMemcpyDeviceToHost);

    // Freeing Memory
    cudaFree(d_matrix);
    cudaFree(d_filter);
    cudaFree(d_ans);

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */
    
    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}

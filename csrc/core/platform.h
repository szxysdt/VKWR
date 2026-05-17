#ifndef VKWR_PLATFORM_H
#define VKWR_PLATFORM_H

#if defined(__CUDA__) || defined(__CUDACC__)
  #define VKWR_CUDA_AVAILABLE 1
  #define VKWR_DEVICE_TYPE "cuda"
#elif defined(__HIP__)
  #define VKWR_HIP_AVAILABLE 1
  #define VKWR_DEVICE_TYPE "hip"
#else
  #define VKWR_CPU_ONLY 1
  #define VKWR_DEVICE_TYPE "cpu"
#endif

#endif // VKWR_PLATFORM_H

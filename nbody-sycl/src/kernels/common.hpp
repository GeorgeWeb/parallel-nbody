#ifndef NBODY_SYCL_KERNELS_COMMON_HPP_
#define NBODY_SYCL_KERNELS_COMMON_HPP_

struct data_access_t {
  static constexpr int global = 1;
  static constexpr int local = 2;
  static constexpr int coalesced = 3;
};

namespace opencl_config {
static constexpr int cache_line = 32;  // best: 32/CPU, 16/GPU
}

#endif  // NBODY_SYCL_KERNELS_COMMON_HPP_

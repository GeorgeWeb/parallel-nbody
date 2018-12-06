#ifndef NBODY_SYCL_KERNELS_COMMON_HPP_
#define NBODY_SYCL_KERNELS_COMMON_HPP_

struct data_access_t {
  static constexpr int global = 1;
  static constexpr int local = 2;
  static constexpr int coalesced = 3;
};

#endif  // NBODY_SYCL_KERNELS_COMMON_HPP_

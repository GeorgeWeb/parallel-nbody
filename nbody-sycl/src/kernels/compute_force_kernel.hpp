#ifndef NBODY_SYCL_COMPUTE_FORCE_KERNEL_HPP_
#define NBODY_SYCL_COMPUTE_FORCE_KERNEL_HPP_

// ...
#include "common.hpp"

// sycl helper abstractions and type aliases
#include <sycl_utils.hpp>
namespace sycl = cl::sycl;

#include <limits>

namespace kernels {

// gravity constant
static constexpr float k_grav = 6.67408f;

template <int access, int num_bodies>
class ComputeForceKernel;

// global memory access partial specialisation definition
template <int num_bodies>
class ComputeForceKernel<data_access_t::global, num_bodies> {
 public:
  // initialises the kernel data
  ComputeForceKernel(const read_accessor_t<sycl::float3, 1> position_ptr,
                     const read_accessor_t<float, 1> mass_ptr,
                     write_accessor_t<sycl::float3, 1> gravity_ptr)
      : m_position_ptr(position_ptr),
        m_mass_ptr(mass_ptr),
        m_gravity_ptr(gravity_ptr) {}

  // defines the kernel computation
  void operator()(sycl::item<1> item) {
    const auto id = item.get_id();

    sycl::float3 force_accumulator(0.0f, 0.0f, 0.0f);

    for (auto n = 0; n < num_bodies; ++n) {
      // calculate distance between bodies
      const sycl::float3 dist = m_position_ptr[n] - m_position_ptr[id];
      const float len = sycl::length(dist);
      if (len > 1.0f) {
        // calculating gravity force direction
        const sycl::float3 direction = sycl::normalize(dist);
        // applying Newton's gravity equation: F = G * m1 * m2 / d^2
        force_accumulator +=
            (k_grav * (m_mass_ptr[id] * m_mass_ptr[n]) / (len * len)) *
            direction;
      }
    }

    m_gravity_ptr[id] = force_accumulator;
  }

 private:
  // global data accessors
  const read_accessor_t<sycl::float3, 1> m_position_ptr;
  const read_accessor_t<float, 1> m_mass_ptr;
  write_accessor_t<sycl::float3, 1> m_gravity_ptr;
};

// local memory access partial specialisation definition
template <int num_bodies>
class ComputeForceKernel<data_access_t::local, num_bodies> {
 public:
  // initialises the kernel data
  ComputeForceKernel(
      read_write_accessor_t<sycl::float3, 1, sycl::access::target::local>
          gravity_scratch_ptr,
      const read_accessor_t<sycl::float3, 1> position_ptr,
      const read_accessor_t<float, 1> mass_ptr,
      write_accessor_t<sycl::float3, 1> gravity_ptr)
      : m_gravity_scratch_ptr(gravity_scratch_ptr),
        m_position_ptr(position_ptr),
        m_mass_ptr(mass_ptr),
        m_gravity_ptr(gravity_ptr) {}

  // defines the kernel computation
  void operator()(sycl::nd_item<1> item) {
    const size_t global_id = item.get_global_id(0);
    const size_t local_id = item.get_local_id(0);

    // gravity force accumulator (shared within the work-group)
    m_gravity_scratch_ptr[local_id] = {0.0f, 0.0f, 0.0f};

    for (size_t offset = 0; offset < num_bodies; offset += local_size) {
      for (size_t n = 0; n < local_size; ++n) {
        // calculate distance between bodies
        const sycl::float3 dist =
            m_position_ptr[n + offset] - m_position_ptr[global_id];
        const float len = sycl::length(dist);
        if (len > 1.0f) {
          // calculating gravity force direction
          const sycl::float3 direction = sycl::normalize(dist);
          // applying Newton's gravity equation: F = G * m1 * m2 / d^2
          m_gravity_scratch_ptr[local_id] +=
              (k_grav * (m_mass_ptr[global_id] * m_mass_ptr[n + offset]) /
               (len * len)) *
              direction;
        }
      }
      // sync work-items within the work-group when writing is over
      item.barrier(sycl::access::fence_space::local_space);
    }
    // sync work-items within the work-group when writing is over
    item.barrier(sycl::access::fence_space::local_space);

    // output the calculated results from the local device memory
    m_gravity_ptr[global_id] = m_gravity_scratch_ptr[local_id];
  }

 private:
  // local (scratch-pad) accessors
  read_write_accessor_t<sycl::float3, 1, sycl::access::target::local>
      m_gravity_scratch_ptr;
  // global data accessors
  const read_accessor_t<sycl::float3, 1> m_position_ptr;
  const read_accessor_t<float, 1> m_mass_ptr;
  write_accessor_t<sycl::float3, 1> m_gravity_ptr;

  static constexpr size_t local_size = num_bodies / 16u;  /// for my GPU
};

}  // namespace kernels

#endif  // NBODY_SYCL_COMPUTE_FORCE_KERNEL_HPP_

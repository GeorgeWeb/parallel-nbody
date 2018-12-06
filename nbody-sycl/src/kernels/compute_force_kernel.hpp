#ifndef NBODY_SYCL_COMPUTE_FORCE_KERNEL_HPP_
#define NBODY_SYCL_COMPUTE_FORCE_KERNEL_HPP_

// ...
#include "common.hpp"

// sycl helper abstractions and type aliases
#include <sycl_utils.hpp>
namespace sycl = cl::sycl;

// gravity constant
static constexpr float k_grav = 6.67408f;

namespace kernels {

template <int access, int num_bodies>
class ComputeForceKernel;

// global memory partial specialisation definition
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
    for (int n = 0; n < num_bodies; ++n) {
      // calculate distance between bodies
      const sycl::float3 dist = m_position_ptr[n] - m_position_ptr[id];
      const float len = sycl::length(dist);
      if (len > 1.0f) {
        // calculating gravity force's direction
        const sycl::float3 direction = sycl::normalize(dist);
        // applying Newton's gravity equation: F = G * m1 * m2 / d^2
        force_accumulator =
            force_accumulator +
            (k_grav * (m_mass_ptr[n] * m_mass_ptr[id]) / (len * len)) *
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

// local memory partial specialisation definition
template <int num_bodies>
class ComputeForceKernel<data_access_t::local, num_bodies> {
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
    for (int n = 0; n < num_bodies; ++n) {
      // calculate distance between bodies
      const sycl::float3 dist = m_position_ptr[n] - m_position_ptr[id];
      const float len = sycl::length(dist);
      if (len > 1.0f) {
        // calculating gravity force's direction
        const sycl::float3 direction = sycl::normalize(dist);
        // applying Newton's gravity equation: F = G * m1 * m2 / d^2
        force_accumulator =
            force_accumulator +
            (k_grav * (m_mass_ptr[n] * m_mass_ptr[id]) / (len * len)) *
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

}  // namespace kernels

#endif  // NBODY_SYCL_COMPUTE_FORCE_KERNEL_HPP_

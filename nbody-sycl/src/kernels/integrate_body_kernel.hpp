#ifndef NBODY_SYCL_INTEGRATE_BODY_KERNEL_HPP_
#define NBODY_SYCL_INTEGRATE_BODY_KERNEL_HPP_

// ...
#include "common.hpp"

// sycl helper abstractions and type aliases
#include <sycl_utils.hpp>
namespace sycl = cl::sycl;

namespace kernels {

template <int access, int num_bodies>
class IntegrateBodyKernel;

// global memory partial specialisation definition
template <int num_bodies>
class IntegrateBodyKernel<data_access_t::global, num_bodies> {
 public:
  // initialises the kernel data
  IntegrateBodyKernel(const read_accessor_t<sycl::float3, 1> gravity_ptr,
                      read_write_accessor_t<sycl::float3, 1> velocity_ptr,
                      read_write_accessor_t<sycl::float3, 1> position_ptr,
                      const float time_step)
      : m_gravity_ptr(gravity_ptr),
        m_velocity_ptr(velocity_ptr),
        m_position_ptr(position_ptr),
        m_time_step(time_step) {}

  // defines the kernel computation
  void operator()(sycl::item<1> item) {
    const auto id = item.get_id();
    m_velocity_ptr[id] += m_gravity_ptr[id] * m_time_step;
    m_position_ptr[id] += m_velocity_ptr[id] * m_time_step;
  }

 private:
  // global data accessors
  const read_accessor_t<sycl::float3, 1> m_gravity_ptr;
  read_write_accessor_t<sycl::float3, 1> m_velocity_ptr;
  read_write_accessor_t<sycl::float3, 1> m_position_ptr;

  // simulation time step
  const float m_time_step;
};

// local memory partial specialisation definition
template <int num_bodies>
class IntegrateBodyKernel<data_access_t::local, num_bodies> {
 public:
  // initialises the kernel data
  IntegrateBodyKernel(
      read_write_accessor_t<sycl::float3, 1, sycl::access::target::local>
          velocity_scratch_ptr,
      read_write_accessor_t<sycl::float3, 1, sycl::access::target::local>
          position_scratch_ptr,
      const read_accessor_t<sycl::float3, 1> gravity_ptr,
      read_write_accessor_t<sycl::float3, 1> velocity_ptr,
      read_write_accessor_t<sycl::float3, 1> position_ptr,
      const float time_step)
      : m_velocity_scratch_ptr(velocity_scratch_ptr),
        m_position_scratch_ptr(position_scratch_ptr),
        m_gravity_ptr(gravity_ptr),
        m_velocity_ptr(velocity_ptr),
        m_position_ptr(position_ptr),
        m_time_step(time_step) {}

  // defines the kernel computation
  void operator()(sycl::nd_item<1> item) {
    const auto global_id = item.get_global_id(0);
    const auto local_id = item.get_local_id(0);

    // read to data local device memory (scratch-pads)
    m_velocity_scratch_ptr[local_id] = m_velocity_ptr[global_id];
    m_position_scratch_ptr[local_id] = m_position_ptr[global_id];

    // integrate the new velocities and positions
    m_velocity_scratch_ptr[local_id] += m_gravity_ptr[global_id] * m_time_step;
    m_position_scratch_ptr[local_id] += m_velocity_ptr[global_id] * m_time_step;

    // sync work-items within the work-group when writing is over
    item.barrier(sycl::access::fence_space::local_space);

    // output the calculated results from the scratch-pads
    m_velocity_ptr[global_id] = m_velocity_scratch_ptr[local_id];
    m_position_ptr[global_id] = m_position_scratch_ptr[local_id];
  }

 private:
  // local (scratch-pad) accessors
  read_write_accessor_t<sycl::float3, 1, sycl::access::target::local>
      m_velocity_scratch_ptr;
  read_write_accessor_t<sycl::float3, 1, sycl::access::target::local>
      m_position_scratch_ptr;
  // global (and constant) data accessors
  const read_accessor_t<sycl::float3, 1> m_gravity_ptr;
  read_write_accessor_t<sycl::float3, 1> m_velocity_ptr;
  read_write_accessor_t<sycl::float3, 1> m_position_ptr;

  // simulation time step
  const float m_time_step;
};

}  // namespace kernels

#endif  // NBODY_SYCL_INTEGRATE_BODY_KERNEL_HPP_

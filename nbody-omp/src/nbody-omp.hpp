#ifndef NBODY_OMP_HPP_
#define NBODY_OMP_HPP_

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

// cross-platform graphics library
#include <graphics.hpp>
namespace gfx = graphics;

// abstract in a header file in the root /include dir.
static float GetRand(float lo, float hi) {
  return lo + static_cast<float>(rand()) /
                  (static_cast<float>(RAND_MAX / (hi - lo)));
}

// gravity constant
static constexpr float k_grav = 6.67408f;

class Body {
 public:
  // default mesh setup constructor resulting in graphics representation is
  // limited to cube meshes at this stage
  Body()
      : m_mesh(std::make_shared<gfx::Mesh>(gfx::Shape::CUBE)),
        m_mass(1.0f),
        m_gravity(glm::vec3(0.0f, -9.8f, 0.0f)) {}

  std::shared_ptr<gfx::Mesh> GetMesh() const { return m_mesh; }

  glm::vec3 GetPosition() const { return m_position; }
  glm::vec3 GetVelocity() const { return m_velocity; }
  glm::vec3 GetAcceleration() const { return m_acceleration; }
  glm::vec3 GetGravity() const { return m_gravity; }
  float GetMass() const { return m_mass; }

  void SetPosition(const glm::vec3 &axis) {
    m_position = axis;
    m_mesh->SetPosition(axis);
  }
  void SetVelocity(const glm::vec3 &axis) { m_velocity = axis; }
  void SetAcceleration(const glm::vec3 &axis) { m_acceleration = axis; }
  void SetGravity(const glm::vec3 &gravity) { m_gravity = gravity; }
  void SetMass(float mass) { m_mass = mass; }

 private:
  std::shared_ptr<gfx::Mesh> m_mesh;

  glm::vec3 m_position;
  // glm::vec3 m_orientation;
  glm::vec3 m_velocity;
  // glm::vec3 m_angular_velocity;
  glm::vec3 m_acceleration;
  float m_mass;

  // gravity force
  glm::vec3 m_gravity;
};

template <int num_bodies>
class NbodyScene {
 public:
  // graphics screen
  gfx::Window window;

  // constructs a scene from a window definition
  explicit NbodyScene(gfx::Window t_window) : window(std::move(t_window)) {}

  void OnLoad() {
    // c-style seeding the rand() generator
    srand(time(0));

    // initialise m_shader program
    m_shader = std::make_unique<gfx::Shader>("shaders/default.vert",
                                           "shaders/default.frag");
    // initialise m_renderer for this scene
    m_renderer = std::make_unique<gfx::Renderer>();
    m_camera = std::make_unique<gfx::Camera>(glm::vec3(0.0f, 0.0f, 500.0f / 4));

    // initiate the body system
    for (int i = 0; i < num_bodies; ++i) {
      // insantiate a body at random position
      auto body = std::make_shared<Body>();
      body->SetPosition(glm::vec3(GetRand(-200.0f / 4, 200.0f / 4),
                                  GetRand(-100.0f / 4, 100.0f / 4),
                                  GetRand(-50.0f / 4, 50.0f / 4)));
      body->SetMass(1.0f);
      m_bodies.push_back(std::move(body));
    }
  }

  void OnUpdate() {
    computeForces();
    integrateBodies();
  }

  void OnDraw() {
    for (int i = 0; i < num_bodies; ++i) {
      m_renderer->Draw(m_shader, m_camera, m_bodies.at(i)->GetMesh());
    }
  }

 private:
 // rendering controllers
  std::unique_ptr<gfx::Camera> m_camera;
  std::unique_ptr<gfx::Shader> m_shader;
  std::unique_ptr<gfx::Renderer> m_renderer;

  // basic physics m_bodies
  std::vector<std::shared_ptr<Body>> m_bodies;
  
  /* the following functions implement the main n-body compuational parts, and
   * are implemented sequentially, though, in a way that allows for parallel
   * optimisation. */

  // inegrates each body's position
  void integrateBodies() {
    // apply eueler's integration for each body's velocity and acceleration
    // ... maybe angular velocity too
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_bodies; ++i) {
      m_bodies.at(i)->SetAcceleration(m_bodies.at(i)->GetGravity());
      m_bodies.at(i)->SetVelocity(
          m_bodies.at(i)->GetVelocity() +
          (m_bodies.at(i)->GetAcceleration() * gfx::delta_time));
      m_bodies.at(i)->SetPosition(
          m_bodies.at(i)->GetPosition() +
          (m_bodies.at(i)->GetVelocity() * gfx::delta_time));
    }
  }

  // implementation is very gravity-specific at this stage
  void computeForces() {
// apply gravitational force to each body's velocity
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_bodies; ++i) {
      glm::vec3 force_accumulator(0.0f);
#pragma omp parallel for schedule(static)
      for (int j = 0; j < num_bodies; ++j) {
        // calculate distance between m_bodies
        const glm::vec3 dist =
            m_bodies.at(j)->GetPosition() - m_bodies.at(i)->GetPosition();
        const float len = glm::length(dist);
        if (len > 1.0f) {
          // calculating gravity force's direction
          const glm::vec3 direction = glm::normalize(dist);
          // applying Newton's gravity equation: F = G * m1 * m2 / d^2
          force_accumulator +=
              (k_grav * (m_bodies.at(j)->GetMass() * m_bodies.at(i)->GetMass()) /
               (len * len)) *
              direction;
        }
      }
      m_bodies.at(i)->SetGravity(force_accumulator);
    }
  }
};

#endif  // NBODY_OMP_HPP_

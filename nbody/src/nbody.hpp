#ifndef NBODY_HPP_
#define NBODY_HPP_

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

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
  float GetMass() const { return m_mass; }

  glm::vec3 GetGravity() const { return m_gravity; }

  void SetPosition(const glm::vec3 &axis) {
    m_position = axis;
    m_mesh->SetPosition(axis);
  }
  void SetVelocity(const glm::vec3 &axis) { m_velocity = axis; }
  void SetMass(float mass) { m_mass = mass; }
  void SetAcceleration(const glm::vec3 &axis) { m_acceleration = axis; }

  void SetGravity(const glm::vec3 &gravity) { m_gravity = gravity; }

 private:
  std::shared_ptr<gfx::Mesh> m_mesh;

  glm::vec3 m_position;
  // glm::vec3 m_orientation;
  glm::vec3 m_velocity;
  // glm::vec3 m_angular_velocity;
  glm::vec3 m_acceleration;
  float m_mass;

  // ... gravity force
  glm::vec3 m_gravity;
};

template <int num_bodies>
class NbodyScene {
 public:
  // replace these 3 with a window instance
  int width;
  int height;
  std::string title;

  std::vector<std::shared_ptr<Body>> bodies;

  std::unique_ptr<gfx::Camera> camera;
  std::unique_ptr<gfx::Shader> shader;
  std::unique_ptr<gfx::Renderer> renderer;

  NbodyScene(int t_width, int t_height, std::string_view t_title)
      : width(t_width), height(t_height), title(t_title) {}

  void OnLoad() {
    // c-style seeding the rand() generator
    srand(time(0));

    // initialise shader program
    shader = std::make_unique<gfx::Shader>("shaders/default.vert",
                                           "shaders/default.frag");
    // initialise renderer for this scene
    renderer = std::make_unique<gfx::Renderer>();
    camera = std::make_unique<gfx::Camera>(glm::vec3(0.0f, 0.0f, 500.0f / 4));

    // initiate the body system
    for (int i = 0; i < num_bodies; ++i) {
      // insantiate a body at random position
      auto body = std::make_shared<Body>();
      body->GetMesh()->Scale(glm::vec3(2.0f));
      body->SetPosition(glm::vec3(GetRand(-200.0f / 4, 200.0f / 4),
                                  GetRand(-100.0f / 4, 100.0f / 4),
                                  GetRand(-50.0f / 4, 50.0f / 4)));
      body->SetMass(1.0f);
      bodies.push_back(std::move(body));
    }
  }

  void OnUpdate() {
    computeForces();
    integrateBodies();
  }

  void OnDraw() {
    for (int i = 0; i < num_bodies; ++i) {
      renderer->Draw(shader, camera, bodies.at(i)->GetMesh());
    }
  }

 private:
  /* the following functions implement the main n-body compuational parts, and
   * are implemented sequentially, though, in a way that allows for parallel
   * optimisation. */

  // implementation is very gravity-specific at this stage
  void computeForces() {
    // apply gravitational force to each body's velocity
    for (int i = 0; i < num_bodies; ++i) {
      glm::vec3 force_accumulator(0.0f);
      for (int j = 0; j < num_bodies; ++j) {
        // calculate distance between bodies
        const glm::vec3 dist =
            bodies.at(j)->GetPosition() - bodies.at(i)->GetPosition();
        const float len = glm::length(dist);
        if (len > 1.0f) {
          // calculating gravity force's direction
          const glm::vec3 direction = glm::normalize(dist);
          // applying Newton's gravity equation: F = G * m1 * m2 / d^2
          force_accumulator +=
              (k_grav * (bodies.at(j)->GetMass() * bodies.at(i)->GetMass()) /
               (len * len)) *
              direction;
        }
      }
      bodies.at(i)->SetGravity(force_accumulator);
    }
  }

  // inegrates each body's position
  void integrateBodies() {
    // apply eueler's integration for each body's velocity and acceleration
    // ... maybe angular velocity too
    for (int i = 0; i < num_bodies; ++i) {
      bodies.at(i)->SetAcceleration(bodies.at(i)->GetGravity());
      bodies.at(i)->SetVelocity(
          bodies.at(i)->GetVelocity() +
          (bodies.at(i)->GetAcceleration() * gfx::delta_time));
      bodies.at(i)->SetPosition(
          bodies.at(i)->GetPosition() +
          (bodies.at(i)->GetVelocity() * gfx::delta_time));
    }
  }
};

#endif  // NBODY_HPP_

#ifndef GRAPHICS_RENDERER_HPP_
#define GRAPHICS_RENDERER_HPP_

#include "../camera/camera.hpp"
#include "../shader/shader.hpp"
#include "mesh.hpp"

#include <iostream>

#include <omp.h>

namespace graphics {

class Renderer {
 public:
  Renderer() { init(); }

  void Draw(const std::unique_ptr<Shader> &shader,
            const std::unique_ptr<Camera> &camera,
            const std::shared_ptr<Mesh> &mesh) {
    // select shader to use on start
    shader->Bind();
    // set system uniforms
    {
      shader->SetMat4("projection", camera->GetProjection());
      shader->SetMat4("view", camera->GetView());
      shader->SetMat4("model", mesh->transform->GetModel());
    }
    // display the mesh on the screen
    mesh->Draw();
  }

 private:
  // do some renderer initialisation
  bool init() const {
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
  }
};

}  // namespace graphics

#endif  // GRAPHICS_RENDERER_HPP_

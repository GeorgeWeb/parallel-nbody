// the nbody simulation scene
#include "nbody.hpp"

static constexpr int num_bodies = 1024 / 4;

// entry point of the nbody application
int main() {
  // scene type alias
  typedef NbodyScene<num_bodies> scene_t;

  // define application
  gfx::Application app;
  // define scene to load
  auto scene = std::make_shared<scene_t>(1280, 720, "n-body app");
  // load the scene (and run the app)
  app.LoadScene(scene);

  return 0;
}

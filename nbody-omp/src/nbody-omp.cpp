// the OpenMP nbody simulation scene
#include "nbody-omp.hpp"

// number of n-body elements
static constexpr int num_bodies = 512;

// graphics window constants
static constexpr int screen_width = 1280;
static constexpr int screen_height = 720;
static constexpr std::string_view screen_title = "#openmp n-body app";

// entry point of the nbody application
int main() {
  // scene type alias
  typedef NbodyScene<num_bodies> scene_t;

  // define application
  gfx::Application app;
  // define scene to load
  auto scene = std::make_shared<scene_t>(
      gfx::Window(screen_width, screen_height, screen_title));
  // load the scene (and run the app)
  app.LoadScene(scene);

  return 0;
}

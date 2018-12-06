#ifndef GRAPHICS_APP_HPP_
#define GRAPHICS_APP_HPP_

#include "../time/time.hpp"
#include "../window/window.hpp"

#include <functional>
#include <sstream>
#include <string>

float graphics::delta_time{0.0f};
float graphics::current_time{0.0f};
float graphics::last_time{0.0f};
float graphics::accumulator{0.0f};

#define PROFLIE_FPS

namespace graphics {

/* description:
 * ... there are two ways to create an application ...
 * */
class Application {
  using app_event_t = std::function<void()>;

 public:
  Application() = default;

  // construct from external window and main loop function definitions
  Application(Window window, const app_event_t &load_event,
              const app_event_t &update_event, const app_event_t &draw_event)
      : m_window(std::move(window)),
        m_on_load(load_event),
        m_on_update(update_event),
        m_on_draw(draw_event) {
    // initialise application systems
    init();
    // start the main loop of the application
    run();
  }

  // provides the ability to create an application from existing scene setup
  template <class Scene>
  void LoadScene(const std::shared_ptr<Scene> &scene) {
    // define the application window
    m_window = std::move(scene->window);
    // assign application main loop functions
    const auto load_event = std::mem_fn(&Scene::OnLoad);
    const auto update_event = std::mem_fn(&Scene::OnUpdate);
    const auto draw_event = std::mem_fn(&Scene::OnDraw);
    m_on_load = std::bind(std::move(load_event), scene);
    m_on_update = std::bind(std::move(update_event), scene);
    m_on_draw = std::bind(std::move(draw_event), scene);
    // initialise application systems
    init();
    // start the main loop of the application
    run();
  }

  Application(const Application &) = default;
  Application &operator=(const Application &) = delete;
  Application(Application &&) = delete;
  Application &operator=(Application &&) = default;
  ~Application() = default;

 private:
  Window m_window;

  // custom/user-defined function behaviour for load
  app_event_t m_on_load;
  // custom/user-defined function behaviour for update
  app_event_t m_on_update;
  // custom/user-defined function behaviour for render
  app_event_t m_on_draw;

  void init() {
    // initialise frame time integration
    delta_time = 0.025f;
    last_time = static_cast<float>(glfwGetTime());
    current_time = static_cast<float>(glfwGetTime());

    // load initial resources and entities
    std::invoke(m_on_load);
  }

  void run() {
    while (!m_window.IsClosed()) {
      tick();
      display();
    }
  }

  void tick() {
    // an accurate frame time integration
    auto new_time = static_cast<float>(glfwGetTime());
    auto frame_time = new_time - current_time;
    frame_time *= 1.0f;
    current_time = new_time;
    accumulator += frame_time;

    while (accumulator >= delta_time) {
      // manage interactions with window
      m_window.PollEvents();

      // update entities (animations)
      std::invoke(m_on_update);

      // reset accumulator
      accumulator -= delta_time;
    }

#ifdef PROFLIE_FPS
    profileFps();
#endif
  }

  void display() {
    // clear the color and depth buffers
    m_window.Clear();

    /// draw entities
    std::invoke(m_on_draw);

    // prepare for next frame
    m_window.Display();
  }

  void profileFps() {
    // initialise frames counter
    static float frame_time = 0.0f;
    static unsigned int frames = 0;

    // update frames counter
    frame_time = current_time - last_time;
    frames++;

    // calculate and display frame data on 1-second interval
    if (frame_time >= 1.0f) {
      // calculate fps and ms/per-frame
      const auto fps = static_cast<float>(frames) / frame_time;
      const auto ms = 1000.0f / fps;  /// ms per frame
      static const auto orig_title = std::string(m_window.GetTitle());

      // change the title on the window to display the fps and ms
      std::ostringstream fps_title;
      fps_title.precision(3);
      fps_title << std::fixed << orig_title << " | fps: " << fps
                << " | ms: " << ms;
      m_window.ChangeTitle(fps_title.str());

      // reset frames counter (and get next second)
      frames = 0;
      last_time += 1.0f;
    }
  }
};

}  // namespace graphics

#endif  // GRAPHICS_APP_HPP_

#ifndef GRAPHICS_WINDOW_HPP_
#define GRAPHICS_WINDOW_HPP_

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

namespace graphics {

namespace detail {
/** @descriton:
 * functor to manage the GLFWwindow resource deallocation
 ** @params:
 * window -
 */
struct WindowDestruct {
  void operator()(GLFWwindow *window) {
    // free GLFWwindow resources
    glfwDestroyWindow(window);
    // free all glfw resources
    glfwTerminate();
  }
};

/** @descriton:
 * custom provided exceptions
 */
struct InitGladException : public std::exception {
  const char *what() const throw() { return "Unable to initialize GLAD"; }
};
struct InitGlfwException : public std::exception {
  const char *what() const throw() { return "Unable to initialize GLFW"; }
};
struct InitWindowException : public std::exception {
  const char *what() const throw() {
    return "Could not create the window object.";
  }
};
}  // namespace detail

class Window final {
  // custom type alias for the window object
  using window_t = std::unique_ptr<GLFWwindow, detail::WindowDestruct>;
  using window_event_t = std::function<void()>;

 public:
  // disable object copying and moving
  Window() = default;
  Window(int width, int height, std::string_view title)
      : m_width{width}, m_height{height}, m_title{title} {
    // attempt to initialize the window
    try {
      initGlfw([=]() {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
      });
      initWindow(width, height, title);
      initGlad();
    }  // handle GLFW initialization exception
    catch (const detail::InitGlfwException &e) {
      std::cerr << e.what() << std::endl;
    }  // handle window_t creation exception
    catch (const detail::InitWindowException &e) {
      std::cerr << e.what() << std::endl;
    }  // handle GLAD initialization exception
    catch (const detail::InitGladException &e) {
      std::cerr << e.what() << std::endl;
    }
  }
  Window(const Window &) = delete;
  Window(Window &&) = default;
  Window &operator=(const Window &) = delete;
  Window &operator=(Window &&) = default;
  ~Window() = default;

  const window_t &GetHandle() const { return m_handle; }

  std::string_view GetTitle() const { return m_title; }

  void ChangeTitle(std::string_view new_title) {
    m_title = new_title;
    resetTitle(m_title);
  }

  void Clear() {
    constexpr std::array<GLfloat, 4> background = {0.0f, 0.0f, 0.0f, 1.0f};
    constexpr GLfloat depth = 1.0f;
    glClearBufferfv(GL_COLOR, 0, background.data());
    glClearBufferfv(GL_DEPTH, 0, &depth);
  }

  void Display() { glfwSwapBuffers(m_handle.get()); }

  void PollEvents() { glfwPollEvents(); }

  bool IsClosed() { return glfwWindowShouldClose(m_handle.get()); }

 private:
  window_t m_handle;
  int m_width;
  int m_height;
  std::string m_title;

  inline void initGlfw(const window_event_t &hints_event) {
    if (const auto glfw_init = glfwInit(); !glfw_init) {
      throw detail::InitGlfwException();
    }
    std::invoke(hints_event);
  }

  inline void initGlad() {
    if (const auto glad_init = gladLoadGLLoader(
            reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
        !glad_init) {
      throw detail::InitGladException();
    }
  }

  inline void initWindow(int width, int height, std::string_view title) {
    // create the window object from GLFWwindow instance
    if (auto window(
            glfwCreateWindow(width, height, title.data(), nullptr, nullptr));
        !window) {
      throw detail::InitWindowException();
    } else {
      m_handle.reset(std::move(window));
    }

    // make the current context
    glfwMakeContextCurrent(m_handle.get());
  }

  inline void resetTitle(std::string_view new_title) {
    glfwSetWindowTitle(m_handle.get(), new_title.data());
  }
};

}  // namespace graphics

#endif  // GRAPHICS_WINDOW_HPP_

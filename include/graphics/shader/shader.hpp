#ifndef GRAPHICS_SHADER_HPP_
#define GRAPHICS_SHADER_HPP_

#include "../util/glutil.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace graphics {

class Shader {
  enum ShaderType { VERTEX, FRAGMENT, PROGRAM };

 public:
  Shader() = default;
  // constructor that generates the shader program from custom custom location
  Shader(const std::string &vertexPath, const std::string &fragmentPath) {
    Load(vertexPath, fragmentPath);
  }

  unsigned int GetHandle() const { return m_handle; }

  void Load(const std::string &vertexPath, const std::string &fragmentPath) {
    const std::string vsString = readShaderFile(vertexPath);
    const std::string fsString = readShaderFile(fragmentPath);

    const GLchar *vsSourcePtr = vsString.c_str();
    const GLchar *fsSourcePtr = fsString.c_str();

    // vertex shader source
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSourcePtr, NULL);
    glCompileShader(vs);

    // check for compilation errors
    checkCompileErrors(vs, VERTEX);

    // fragment shader source
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSourcePtr, NULL);
    glCompileShader(fs);

    // check for compilation errors
    checkCompileErrors(fs, FRAGMENT);

    // create shader program
    m_handle = glCreateProgram();
    glAttachShader(m_handle, vs);
    glAttachShader(m_handle, fs);
    glLinkProgram(m_handle);

    checkCompileErrors(m_handle, PROGRAM);

    // the shaders are linked to a program, so the shaders themselves are no
    // longer required.
    glDeleteShader(vs);
    glDeleteShader(fs);
  }

  // activate the shader
  void Bind() const {
    if (m_handle > 0) {
      glUseProgram(m_handle);
    }
  }

  void Unbind() const { glUseProgram(0); }

  int GetUniformLocation(const std::string &name) const {
    return glGetUniformLocation(m_handle, name.c_str());
  }

  void SetBool(const std::string &name, bool value) const {
    glUniform1i(GetUniformLocation(name), static_cast<int>(value));
  }
  void SetInt(const std::string &name, int value) const {
    glUniform1i(GetUniformLocation(name), value);
  }
  void SetFloat(const std::string &name, float value) const {
    glUniform1f(GetUniformLocation(name), value);
  }
  void SetVec2(const std::string &name, const glm::vec2 &vect) const {
    glUniform2fv(GetUniformLocation(name), 1, glm::value_ptr(vect));
  }
  void SetVec2(const std::string &name, float x, float y) const {
    glUniform2f(GetUniformLocation(name), x, y);
  }
  void SetVec3(const std::string &name, const glm::vec3 &vect) const {
    glUniform3fv(GetUniformLocation(name), 1, glm::value_ptr(vect));
  }
  void SetVec3(const std::string &name, float x, float y, float z) const {
    glUniform3f(GetUniformLocation(name), x, y, z);
  }
  void SetVec4(const std::string &name, const glm::vec4 &vect) const {
    glUniform4fv(GetUniformLocation(name), 1, glm::value_ptr(vect));
  }
  void SetVec4(const std::string &name, float x, float y, float z, float w) {
    glUniform4f(GetUniformLocation(name), x, y, z, w);
  }
  void SetMat2(const std::string &name, const glm::mat2 &mat) const {
    glUniformMatrix2fv(GetUniformLocation(name), 1, GL_FALSE, glm::value_ptr(mat));
  }
  void SetMat3(const std::string &name, const glm::mat3 &mat) const {
    glUniformMatrix3fv(GetUniformLocation(name), 1, GL_FALSE, glm::value_ptr(mat));
  }
  void SetMat4(const std::string &name, const glm::mat4 &mat) const {
    glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, glm::value_ptr(mat));
  }

 private:
  // utility function for checking shader compilation/linking errors.
  void checkCompileErrors(GLuint shader, ShaderType type) {
    int status = 0;

    if (type == PROGRAM) {
      glGetProgramiv(m_handle, GL_LINK_STATUS, &status);
      if (status == GL_FALSE) {
        GLint length;
        glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &length);

        std::string errorLog(m_handle, ' ');
        glGetProgramInfoLog(m_handle, length, &length, &errorLog[0]);
        std::cerr << "Error! Program failed to link. " << errorLog << std::endl;
      }
    } else {  // VERTEX or FRAGMENT
      glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
      if (status == GL_FALSE) {
        GLint length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

        std::string errorLog(length, ' ');
        glGetShaderInfoLog(shader, length, &length, &errorLog[0]);
        std::cerr << "Error! Shader failed to compile. (" << type
                  << "): " << errorLog << std::endl;
      }
    }
  }

 private:
  unsigned int m_handle;

  std::string readShaderFile(std::string_view filename) {
    std::stringstream ss;
    try {
      std::ifstream file;
      file.open(std::string(filename), std::ios::in);
      if (!file.fail()) {
        ss << file.rdbuf();
      }
      file.close();
    } catch (const std::exception &e) {
      std::cerr << "ERROR: Cannot read the file: " << filename << " | "
                << e.what() << std::endl;
    }
    return ss.str();
  }
};

}  // namespace graphics

#endif  // GRAPHICS_SHADER_HPP_
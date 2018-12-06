#ifndef FILE_IO_HPP_
#define FILE_IO_HPP_

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

namespace file_io {

enum class TextFormatting : int {
  NONE = 0,
  COMA = 1,
  INTERVAL = 2,
  TABULATION = 3,
  BREAK_LINE = 4,
};

enum class WriteMode : int {
  NORMAL = 0,
  OVERWRITE = 1,
  APPEND = 2,
  APPEND_ONCE = 3
};

std::string ReadFileAsStr(std::string_view filename,
                          TextFormatting formatting = TextFormatting::NONE) {
  std::stringstream ss;
  try {
    std::ifstream file;
    file.open(std::string(filename), std::ios::in);
    if (!file.fail()) {
      // TODO: implement each case specifically
      switch (formatting) {
        default:
        case TextFormatting::NONE:
          ss << file.rdbuf();
          break;
        case TextFormatting::COMA:
          ss << file.rdbuf();
          break;
        case TextFormatting::INTERVAL:
          ss << file.rdbuf();
          break;
        case TextFormatting::TABULATION:
          ss << file.rdbuf();
          break;
        case TextFormatting::BREAK_LINE:
          ss << file.rdbuf();
          break;
      }
    }
    file.close();
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Cannot read the file: " << filename << " | "
              << e.what() << std::endl;
  }
  return ss.str();
}

template <typename T>
void WriteStrToFile(WriteMode mode, T data, std::string_view filename,
                    TextFormatting formatting = TextFormatting::NONE) {
  try {
    std::ofstream file;
    switch (mode) {
      default:
      case WriteMode::NORMAL:
        file.open(std::string(filename), std::ios::out);
        break;
      case WriteMode::OVERWRITE:
        file.open(std::string(filename), std::ios::trunc);
        break;
      case WriteMode::APPEND:
        file.open(std::string(filename), std::ios::app);
        break;
      case WriteMode::APPEND_ONCE:
        file.open(std::string(filename), std::ios::ate);
        break;
    }
    // set max precision if the data type is floating point (float or double)
    if constexpr (std::is_floating_point_v<T>) {
      file.precision(std::numeric_limits<T>::max_digits10);
    }
    switch (formatting) {
      default:
      case TextFormatting::NONE:
        file << data;
        break;
      case TextFormatting::COMA:
        file << data << ',';
        break;
      case TextFormatting::INTERVAL:
        file << data << ' ';
        break;
      case TextFormatting::TABULATION:
        file << data << '\t';
        break;
      case TextFormatting::BREAK_LINE:
        file << data << '\n';
        break;
    }
    file.close();
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Cannot open the file: " << filename << " | "
              << e.what() << std::endl;
  }
}

}  // namespace file_io

#endif  // FILE_IO_HPP_

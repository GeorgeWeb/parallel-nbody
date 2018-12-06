#ifndef UTIL_TIMER_HPP_
#define UTIL_TIMER_HPP_

#include <chrono>
#include <ctime>
#include <iostream>
#include <string>
#include <type_traits>

namespace timer {

// prototype of the Timer class
template <typename T, typename Ratio>
class Timer;

// prototype of the operator to be overloaded for class Timer
template <typename T, typename Ratio>
std::ostream &operator<<(std::ostream &os, const Timer<T, Ratio> &obj);

// implementing the Timer class
template <typename T = double, typename Ratio = std::ratio<1, 1>>
class Timer final {
  // convenience type aliases
  using clock = std::chrono::high_resolution_clock;
  using duration = std::chrono::duration<T, Ratio>;

 public:
  Timer() : m_initial_tp(clock::now()) {}
  ~Timer() = default;

  inline void Reset() { m_initial_tp = clock::now(); }

  inline T GetElapsedTime() const {
    duration elapsed = clock::now() - m_initial_tp;
    return elapsed.count();
  }

  inline std::string RatioToString() const {
    return std::string(std::is_same<Ratio, std::ratio<1, 1>>::value
                           ? "second(s)"
                           : std::is_same<Ratio, std::milli>::value
                                 ? "millisecond(s)"
                                 : std::is_same<Ratio, std::micro>::value
                                       ? "microsecond(s)"
                                       : std::is_same<Ratio, std::nano>::value
                                             ? "nanosecond(s)"
                                             : "");
  }

  // declaring operator<< overload for class Timer
  friend std::ostream &operator<<<>(std::ostream &os, const Timer &obj);

 private:
  std::chrono::time_point<clock> m_initial_tp;
};

// implementing the operator<< overload for class Timer
template <typename T, typename Ratio>
std::ostream &operator<<(std::ostream &os, const Timer<T, Ratio> &obj) {
  return os << obj.GetElapsedTime() << ' ' << obj.RatioToString();
}

}  // namespace timer

#endif  // UTIL_TIMER_HPP_

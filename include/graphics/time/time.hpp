#ifndef GRAPHICS_TIME_HPP_
#define GRAPHICS_TIME_HPP_

namespace graphics {

extern float delta_time;
extern float current_time;
extern float last_time;
extern float accumulator;
// counts the total number of time steps for the duration of the application
extern int time_step_count;

}  // namespace graphics

#endif  // GRAPHICS_TIME_HPP_
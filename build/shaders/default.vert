#version 330

layout (location = 0) in vec3 position;

out vec3 vposition;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main() {
    // output position
    vposition = position;
    vec3 world_position = vec3(model * vec4(position, 1.0));
    gl_Position = projection * view * vec4(world_position, 1.0);
}

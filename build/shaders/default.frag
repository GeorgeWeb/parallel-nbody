#version 330

in vec3 vposition;

out vec4 color;

void main() {
    // output color
    vec3 albedo = clamp(vposition, 0.0f, 1.0f);
    color = vec4(albedo, 1.0f);
}

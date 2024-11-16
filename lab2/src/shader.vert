#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

layout(binding = 0) uniform UniformBufferObject {
    mat4 mvp;
} ubo;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec2 tex_coords;

void main() {
    gl_Position = ubo.mvp * vec4(in_position, 1.0);
    frag_color = vec3(1.0, 0.0, 0.0);
    tex_coords = in_uv;
}

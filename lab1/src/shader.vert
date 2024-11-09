#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(binding = 0) uniform UniformBufferObject {
    vec3 pos;
} ubo;

layout(location = 0) out vec3 frag_color;

void main() {
    gl_Position = vec4(in_position + ubo.pos, 1.0);
    frag_color = in_color;
}

#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(binding = 0) uniform UniformBufferObject {
    mat4 mvp;
} ubo;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec2 tex_coords;

void main() {
    vec4 pos = ubo.mvp * vec4(in_position, 1.0);
    gl_Position = pos;
    frag_color = log(vec3(in_position.z, in_position.z, in_position.z)) / 10.0;
    tex_coords = vec2(in_uv.x, 1.0 - in_uv.y);
}

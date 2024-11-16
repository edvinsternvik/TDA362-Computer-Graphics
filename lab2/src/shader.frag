#version 450

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec2 in_texture_coords;

layout(binding = 1) uniform sampler2D texture_sampler;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(texture(texture_sampler, in_texture_coords).rgb, 1.0);
}

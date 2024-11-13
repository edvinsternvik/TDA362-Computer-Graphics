#version 450


layout(location = 0) in vec3 in_color;
layout(location = 1) in vec2 in_texture_coordinates;

layout(binding = 1) uniform sampler2D texture_sampler;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 texture_color = texture(texture_sampler, in_texture_coordinates).rgb;
    out_color = vec4(texture_color * in_color, 1.0);
}

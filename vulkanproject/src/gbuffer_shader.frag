#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(normalize(in_normal), length(in_position));
}

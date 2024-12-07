#version 450

layout(location = 0) in vec2 in_position;

layout(location = 0) out vec2 out_uv;

void main() {
    out_uv = 0.5 * vec2(1.0 + in_position.x, 1.0 + in_position.y);

    gl_Position = vec4(in_position, 0.0, 1.0);
}

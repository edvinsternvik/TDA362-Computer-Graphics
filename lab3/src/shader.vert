#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model_matrix;
    mat4 model_view_matrix;
    mat4 model_view_projection_matrix;
} ubo;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 tex_coords;

void main() {
    vec4 pos = ubo.model_view_projection_matrix * vec4(in_position, 1.0);
    gl_Position = pos;
    out_normal = vec3(ubo.model_matrix * vec4(in_normal, 0.0));
    tex_coords = vec2(in_uv.x, 1.0 - in_uv.y);
}

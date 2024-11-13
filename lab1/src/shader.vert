#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec2 in_texture_coordinates;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model_view_projection_matrix;
} ubo;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec2 frag_texture_coordinates;

void main() {
    gl_Position = ubo.model_view_projection_matrix * vec4(in_position, 1.0);
    frag_color = in_color;
    frag_texture_coordinates = in_texture_coordinates;
}

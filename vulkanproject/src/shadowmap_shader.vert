#version 450

layout(location = 0) in vec3 in_position;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model_matrix;
    mat4 model_view_matrix;
    mat4 model_view_projection_matrix;
};

void main() {
    gl_Position = model_view_projection_matrix * vec4(in_position, 1.0);
}

#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model_matrix;
    mat4 model_view_matrix;
    mat4 model_view_projection_matrix;
};

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;

void main() {
    vec4 vs_position = model_view_matrix * vec4(in_position, 1.0);
    out_position = vs_position.xyz / vs_position.w;
    mat4 normal_matrix = transpose(inverse(model_view_matrix));
    out_normal = vec3(normal_matrix * vec4(in_normal, 0.0));
    gl_Position = model_view_projection_matrix * vec4(in_position, 1.0);
}

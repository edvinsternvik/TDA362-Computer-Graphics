#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model_matrix;
    mat4 model_view_matrix;
    mat4 model_view_projection_matrix;
} ubo;

layout(set = 1, binding = 0) uniform GlobalUBO {
    mat4 light_matrix;
    mat4 view_inverse;
    vec3 light_view_pos;
    float light_intensity;
    vec3 light_color;
    float env_multiplier;
    vec3 light_view_dir;
    float spot_inner_angle;
    float spot_outer_angle;
    bool use_spot_light;
    bool use_soft_falloff;
};

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_uv;
layout(location = 3) out vec4 out_shadow_map_coords;

void main() {
    vec4 pos = ubo.model_view_projection_matrix * vec4(in_position, 1.0);
    out_position = vec3(ubo.model_view_matrix * vec4(in_position, 1.0));

    mat4 normal_matrix = transpose(inverse(ubo.model_view_matrix));
    out_normal = vec3(normal_matrix * vec4(in_normal, 0.0));

    out_uv = vec2(in_uv.x, 1.0 - in_uv.y);

    out_shadow_map_coords = light_matrix * vec4(out_position, 1.0);

    gl_Position = pos;
}

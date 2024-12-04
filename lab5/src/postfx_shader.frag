#version 450

layout(location = 0) in vec2 in_uv;

layout(binding = 0) uniform PostFXUBO {
    mat4 inv_pv;
    vec3 camera_pos;
    float env_multiplier;
};
layout(binding = 1) uniform sampler2D frame_sampler;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = texture(frame_sampler, in_uv);
}

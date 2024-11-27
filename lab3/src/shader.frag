#version 450

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_texture_coords;

layout(binding = 1) uniform MaterialUBO {
    vec3 color;
    float metalic;
    float fresnel;
    float roughness;
    float transparency;
    float ior;
    vec3 emission;
} ubo_material;
layout(binding = 2) uniform sampler2D color_sampler;
layout(binding = 3) uniform sampler2D metalic_sampler;
layout(binding = 4) uniform sampler2D fresnel_sampler;
layout(binding = 5) uniform sampler2D roughness_sampler;
layout(binding = 6) uniform sampler2D emission_sampler;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 normal = normalize(in_normal);

    vec4 color = vec4(ubo_material.color, 1.0);
    vec4 tex_color = texture(color_sampler, in_texture_coords);
    if(tex_color.a != 0.0) {
        color = tex_color;
    }
    vec4 emission = vec4(ubo_material.emission, 0.0);
    vec4 tex_emission = texture(emission_sampler, in_texture_coords);
    if(tex_emission.a != 0.0) {
        emission = tex_emission;
    }

	const vec3 light_dir = normalize(vec3(-0.74, -1, 0.68));
	out_color = vec4(color.rgb * max(dot(normal, -light_dir), 0.3) + emission.rgb, 1.0);
}

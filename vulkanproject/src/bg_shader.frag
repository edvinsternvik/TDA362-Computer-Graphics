#version 450

layout(location = 0) in vec2 in_uv;

layout(binding = 0) uniform BgUBO {
    mat4 inv_pv;
    vec3 camera_pos;
    float env_multiplier;
};
layout(binding = 1) uniform sampler2D bg_sampler;

layout(location = 0) out vec4 out_color;

#define PI 3.14159265359

void main() {
	// Calculate the world-space position of this fragment on the near plane
	vec4 pixel_world_pos = inv_pv * vec4(in_uv * 2.0 - 1.0, 1.0, 1.0);
	pixel_world_pos = (1.0 / pixel_world_pos.w) * pixel_world_pos;

	// Calculate the world-space direction from the camera to that position
	vec3 dir = normalize(pixel_world_pos.xyz - camera_pos);

	// Calculate the spherical coordinates of the direction
	float theta = acos(max(-1.0f, min(1.0f, dir.y)));
	float phi = atan(dir.z, dir.x);
	if(phi < 0.0f) {
		phi = phi + 2.0f * PI;
	}

	// Use these to lookup the color in the environment map
	vec2 lookup = vec2(phi / (2.0 * PI), 1 - theta / PI);
    lookup.y *= -1.0;
	out_color = env_multiplier * texture(bg_sampler, lookup);
    // out_color = vec4(in_uv, 0.0, 1.0);
}

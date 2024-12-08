#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

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

layout(set = 1, binding = 0) uniform GlobalUBO {
    mat4 light_matrix;
    mat4 view_inverse;
    vec3 light_view_pos;
    float light_intensity;
    vec3 light_color;
    float env_multiplier;
};
layout(set = 1, binding = 1) uniform sampler2D env_sampler;
layout(set = 1, binding = 2) uniform sampler2D irradiance_sampler;
layout(set = 1, binding = 3) uniform sampler2D reflection_sampler;
layout(set = 1, binding = 4) uniform sampler2D shadowmap_sampler;

layout(location = 0) out vec4 out_color;

#define PI 3.14159265359

vec3 direct_illumination(vec3 wo, vec3 n, vec3 base_color) {
	///////////////////////////////////////////////////////////////////////////
	// Task 1.2 - Calculate the radiance Li from the light, and the direction
	//            to the light. If the light is backfacing the triangle,
	//            return vec3(0);
	///////////////////////////////////////////////////////////////////////////
    vec3 wi = normalize(light_view_pos - in_position);
    float ndotwo = max(0.0001, dot(n, wo));
    float ndotwi = max(0.0001, dot(n, wi));

    float d = length(light_view_pos - in_position);
    vec3 Li = light_intensity * light_color / (d * d);

    if(dot(wi, n) <= 0.0) {
        return vec3(0.0);
    }

	///////////////////////////////////////////////////////////////////////////
	// Task 1.3 - Calculate the diffuse term and return that as the result
	///////////////////////////////////////////////////////////////////////////
    vec3 diffuse_term = base_color * (1.0 / PI) * ndotwi * Li;

	///////////////////////////////////////////////////////////////////////////
	// Task 2 - Calculate the Torrance Sparrow BRDF and return the light
	//          reflected from that instead
	///////////////////////////////////////////////////////////////////////////
    vec3 wh = normalize(wi + wo);
    float ndotwh = max(0.0001, dot(n, wh));
    float wodotwh = max(0.0001, dot(wo, wh));
    float widotwh = max(0.0001, dot(wi, wh));

    float R0 = ubo_material.fresnel;
    float F = R0 + (1.0 - R0) * pow((1.0 - widotwh), 5.0);

    float s = ubo_material.roughness;
    float D = (s + 2) * pow(ndotwh, s) / (2 * PI);

    float G0 = 2 * ndotwh * ndotwo / wodotwh;
    float G1 = 2 * ndotwh * ndotwi / wodotwh;
    float G = min(1.0, min(G0, G1));

    float brdf = F * D * G / clamp(4.0 * dot(n, wo) * dot(n, wi), 0.0001, 1.0);

	///////////////////////////////////////////////////////////////////////////
	// Task 3 - Make your shader respect the parameters of our material model.
	///////////////////////////////////////////////////////////////////////////
    vec3 dielectric_term = brdf * ndotwi * Li + (1.0 - F) * diffuse_term;
    vec3 metal_term = brdf * base_color * ndotwi * Li;

    float m = ubo_material.metalic;
	return m * metal_term + (1.0 - m) * dielectric_term;
}

vec3 indirect_illumination(vec3 wo, vec3 n, vec3 base_color) {
	vec3 indirect_illum = vec3(0.f);
	///////////////////////////////////////////////////////////////////////////
	// Task 5 - Lookup the irradiance from the irradiance map and calculate
	//          the diffuse reflection
	///////////////////////////////////////////////////////////////////////////
	vec3 world_normal = vec3(view_inverse * vec4(n, 0.0));
	float theta = acos(max(-1.0f, min(1.0f, world_normal.y)));
	float phi = atan(world_normal.z, world_normal.x);
	if(phi < 0.0f) phi = phi + 2.0f * PI;
	vec2 lookup = vec2(phi / (2.0 * PI), 1 - theta / PI);
    lookup.y *= -1.0;
	vec3 Li = env_multiplier * texture(irradiance_sampler, lookup).rgb;
	vec3 diffuse_term = base_color * (1.0 / PI) * Li;
	indirect_illum = diffuse_term;

	///////////////////////////////////////////////////////////////////////////
	// Task 6 - Look up in the reflection map from the perfect specular
	//          direction and calculate the dielectric and metal terms.
	///////////////////////////////////////////////////////////////////////////
	vec3 wi = normalize(reflect(-wo, n));
	vec3 wr = normalize(vec3(view_inverse * vec4(wi, 0.0)));
	theta = acos(max(-1.0f, min(1.0f, wr.y)));
	phi = atan(wr.z, wr.x);
	if(phi < 0.0f) phi = phi + 2.0f * PI;
	lookup = vec2(phi / (2.0 * PI), 1 - theta / PI);
    lookup.y *= -1.0;
	float roughness = sqrt(sqrt(2.0 / (ubo_material.roughness + 2.0)));
	Li = env_multiplier * textureLod(reflection_sampler, lookup, roughness * 7.0).rgb;
	vec3 wh = normalize(wi + wo);
	float wodotwh = max(0.0, dot(wo, wh));
	float F = ubo_material.fresnel + (1.0 - ubo_material.fresnel) * pow(1.0 - wodotwh, 5.0);
	vec3 dielectric_term = F * Li + (1.0 - F) * diffuse_term;
	vec3 metal_term = F * base_color * Li;

	vec3 microfacet_term = ubo_material.metalic * metal_term + (1.0 - ubo_material.metalic) * dielectric_term;

	indirect_illum = microfacet_term;

	return indirect_illum;
}

void main() {
    vec3 normal = normalize(in_normal);

    vec3 wo = normalize(-in_position);

    vec3 base_color = ubo_material.color;
    vec4 tex_color = texture(color_sampler, in_uv);
    if(tex_color.a != 0.0) {
        base_color *= tex_color.rgb;
    }

    vec4 shadowmap_coords = light_matrix * vec4(in_position, 1.0);
    float depth = texture(shadowmap_sampler, shadowmap_coords.xy / shadowmap_coords.w).x;
    float visibility = (depth >= (shadowmap_coords.z / shadowmap_coords.w)) ? 1.0 : 0.0;

    vec3 dir_elim_term = visibility * direct_illumination(wo, normal, base_color);

    vec3 emission_term = ubo_material.emission;
    vec4 tex_emission = texture(emission_sampler, in_uv);
    if(tex_emission.a != 0.0) {
        emission_term *= tex_emission.rgb;
    }

    vec3 indir_elim_term = indirect_illumination(wo, normal, base_color);

	out_color = vec4(dir_elim_term + emission_term + indir_elim_term, 1.0);

	if(any(isnan(out_color))) {
		out_color.rgb = vec3(1.f, 0.f, 1.f);
	}
}

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

layout(binding = 7) uniform LightUBO {
    vec3 view_position;
} ubo_light;

layout(location = 0) out vec4 out_color;

#define PI 3.14159265359

vec3 direct_illumination(vec3 wo, vec3 n, vec3 base_color) {
	///////////////////////////////////////////////////////////////////////////
	// Task 1.2 - Calculate the radiance Li from the light, and the direction
	//            to the light. If the light is backfacing the triangle,
	//            return vec3(0);
	///////////////////////////////////////////////////////////////////////////
    vec3 wi = normalize(ubo_light.view_position - in_position);
    float ndotwo = max(0.0001, dot(n, wo));
    float ndotwi = max(0.0001, dot(n, wi));

    float d = length(ubo_light.view_position - in_position);
    vec3 Li = 50.0f * vec3(1.0, 1.0, 1.0) / (d * d);

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

	///////////////////////////////////////////////////////////////////////////
	// Task 6 - Look up in the reflection map from the perfect specular
	//          direction and calculate the dielectric and metal terms.
	///////////////////////////////////////////////////////////////////////////

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

    vec3 dir_elim_term = direct_illumination(wo, normal, base_color);

    vec3 emission_term = ubo_material.emission;
    vec4 tex_emission = texture(emission_sampler, in_uv);
    if(tex_emission.a != 0.0) {
        emission_term *= tex_emission.rgb;
    }

	out_color = vec4(dir_elim_term + emission_term, 1.0);

	if(any(isnan(out_color))) {
		out_color.rgb = vec3(1.f, 0.f, 1.f);
	}
}

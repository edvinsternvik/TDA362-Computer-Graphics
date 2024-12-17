#include "material.hpp"
#include "pathtracer.hpp"
#include "sampling.hpp"

namespace pathtracer {

WiSample sample_hemisphere_cosine(const vec3& wo, const vec3& n) {
	mat3 tbn = tangent_space(n);
	vec3 sample = cosine_sample_hemisphere();
	WiSample r;
	r.wi = tbn * sample;
	if(dot(r.wi, n) > 0.0f)
		r.pdf = max(0.0f, dot(r.wi, n)) / M_PI;
	return r;
}

///////////////////////////////////////////////////////////////////////////
// A Lambertian (diffuse) material
///////////////////////////////////////////////////////////////////////////
vec3 Diffuse::f(const vec3& wi, const vec3& wo, const vec3& n) const {
	if(dot(wi, n) <= 0.0f)
		return vec3(0.0f);
	if(!same_hemisphere(wi, wo, n))
		return vec3(0.0f);
	return (1.0f / M_PI) * color;
}

WiSample Diffuse::sample_wi(const vec3& wo, const vec3& n) const {
	WiSample r = sample_hemisphere_cosine(wo, n);
	r.f = f(r.wi, wo, n);
	return r;
}

vec3 MicrofacetBRDF::f(const vec3& wi, const vec3& wo, const vec3& n) const {
    vec3 wh = normalize(wi + wo);

    float n_dot_wh = dot(n, wh);
    float n_dot_wi = dot(n, wi);
    float n_dot_wo = dot(n, wo);
    float wi_dot_wo = dot(wi, wo);
    float wo_dot_wh = dot(wo, wh);

    float D = (shininess + 2.0) * pow(n_dot_wh, shininess)
        / (2.0 * pi<float>());
    float G = min(
        1.0,
        (2.0 * n_dot_wh / wo_dot_wh) * min(n_dot_wo, n_dot_wh)
    );
	return vec3(D * G / (4.0 * n_dot_wo * n_dot_wi));
}

WiSample MicrofacetBRDF::sample_wi(const vec3& wo, const vec3& n) const {
	WiSample r = sample_hemisphere_cosine(wo, n);
	r.f = f(r.wi, wo, n);

	return r;
}


float BSDF::fresnel(const vec3& wi, const vec3& wo) const {
    vec3 wh = normalize(wi + wo);
	return R0 + (1.0 - R0) * pow(1.0 - dot(wh, wi), 5.0);
}


vec3 DielectricBSDF::f(const vec3& wi, const vec3& wo, const vec3& n) const {
    float F = fresnel(wi, wo);
    vec3 BRDF = reflective_material->f(wi, wo, n);
    vec3 BTDF = transmissive_material->f(wi, wo, n);
	return F * BRDF + (1.0f - F) * BTDF;
}

WiSample DielectricBSDF::sample_wi(const vec3& wo, const vec3& n) const {
	WiSample r;

	r = sample_hemisphere_cosine(wo, n);
	r.f = f(r.wi, wo, n);

	return r;
}

vec3 MetalBSDF::f(const vec3& wi, const vec3& wo, const vec3& n) const {
    float F = fresnel(wi, wo);
    vec3 BRDF = reflective_material->f(wi, wo, n);
	return F * BRDF * color;
}

WiSample MetalBSDF::sample_wi(const vec3& wo, const vec3& n) const {
	WiSample r;
	r = sample_hemisphere_cosine(wo, n);
	r.f = f(r.wi, wo, n);
	return r;
}


vec3 BSDFLinearBlend::f(const vec3& wi, const vec3& wo, const vec3& n) const {
    return w * bsdf0->f(wi, wo, n) + (1.0f - w) * bsdf1->f(wi, wo, n);
}

WiSample BSDFLinearBlend::sample_wi(const vec3& wo, const vec3& n) const {
	return WiSample{};
}


#if SOLUTION_PROJECT == PROJECT_REFRACTIONS
///////////////////////////////////////////////////////////////////////////
// A perfect specular refraction.
///////////////////////////////////////////////////////////////////////////
vec3 GlassBTDF::f(const vec3& wi, const vec3& wo, const vec3& n) const {
	if(same_hemisphere(wi, wo, n)) {
		return vec3(0);
	}
	else {
		return vec3(1);
	}
}

WiSample GlassBTDF::sample_wi(const vec3& wo, const vec3& n) const {
	WiSample r;

	float eta;
	glm::vec3 N;
	if(dot(wo, n) > 0.0f) {
		N = n;
		eta = 1.0f / ior;
	}
	else {
		N = -n;
		eta = ior;
	}

	// Alternatively:
	// d = dot(wo, N)
	// k = d * d (1 - eta*eta)
	// wi = normalize(-eta * wo + (d * eta - sqrt(k)) * N)

	// or

	// d = dot(n, wo)
	// k = 1 - eta*eta * (1 - d * d)
	// wi = - eta * wo + ( eta * d - sqrt(k) ) * N

	float w = dot(wo, N) * eta;
	float k = 1.0f + (w - eta) * (w + eta);
	if(k < 0.0f) {
		// Total internal reflection
		r.wi = reflect(-wo, n);
	}
	else {
		k = sqrt(k);
		r.wi = normalize(-eta * wo + (w - k) * N);
	}
	r.pdf = abs(dot(r.wi, n));
	r.f = vec3(1.0f, 1.0f, 1.0f);

	return r;
}

vec3 BTDFLinearBlend::f(const vec3& wi, const vec3& wo, const vec3& n) const {
	return w * btdf0->f(wi, wo, n) + (1.0f - w) * btdf1->f(wi, wo, n);
}

WiSample BTDFLinearBlend::sample_wi(const vec3& wo, const vec3& n) const {
	if(randf() < w) {
		WiSample r = btdf0->sample_wi(wo, n);
		return r;
	}
	else {
		WiSample r = btdf1->sample_wi(wo, n);
		return r;
	}
}

#endif
} // namespace pathtracer

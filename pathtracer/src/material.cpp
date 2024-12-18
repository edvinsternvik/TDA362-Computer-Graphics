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

bool bad_sample(const vec3& wi, const vec3& wo, const vec3& n) {
    return !same_hemisphere(wi, wo, n) || dot(wi, n) <= 0.0;
}

///////////////////////////////////////////////////////////////////////////
// A Lambertian (diffuse) material
///////////////////////////////////////////////////////////////////////////
vec3 Diffuse::f(const vec3& wi, const vec3& wo, const vec3& n) const {
    if(bad_sample(wi, wo, n)) return zero<vec3>();
	return (1.0f / M_PI) * color;
}

WiSample Diffuse::sample_wi(const vec3& wo, const vec3& n) const {
	WiSample r = sample_hemisphere_cosine(wo, n);
	r.f = f(r.wi, wo, n);
	return r;
}

vec3 MicrofacetBRDF::f(const vec3& wi, const vec3& wo, const vec3& n) const {
    if(bad_sample(wi, wo, n)) return zero<vec3>();

    vec3 wh = normalize(wi + wo);

    float n_dot_wh = dot(n, wh);
    float n_dot_wi = dot(n, wi);
    float n_dot_wo = dot(n, wo);
    float wi_dot_wo = dot(wi, wo);
    float wo_dot_wh = dot(wo, wh);

    float D = (shininess + 2.0) * pow(n_dot_wh, shininess)
        / (2.0 * M_PI);
    float G = min(
        1.0,
        (2.0 * n_dot_wh / wo_dot_wh) * min(n_dot_wo, n_dot_wh)
    );
    if(4.0 * n_dot_wo * n_dot_wi == 0.0) {
        return zero<vec3>();
    }
    return vec3(D * G / (4.0 * n_dot_wo * n_dot_wi));
}

WiSample MicrofacetBRDF::sample_wi(const vec3& wo, const vec3& n) const {
    float phi = 2.0f * M_PI * randf();
    float cos_theta = pow(randf(), 1.0f / (shininess + 1));
    float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    vec3 wh = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    wh = normalize(tangent_space(n) * wh);
    float p_wh = (shininess + 1.0) * pow(dot(n, wh), shininess) / (2.0 * M_PI);

    vec3 wi = -normalize(reflect(wo, wh));
    float p_wi = 0.0;
    if(dot(wo, wh) != 0.0) p_wi = p_wh / (4.0 * dot(wo, wh));

    WiSample r;
    r.wi = wi;
    r.pdf = p_wi;
	r.f = f(r.wi, wo, n);

	return r;
}


float BSDF::fresnel(const vec3& wi, const vec3& wo) const {
    vec3 wh = normalize(wi + wo);
	return R0 + (1.0 - R0) * pow(1.0 - dot(wh, wi), 5.0);
}


vec3 DielectricBSDF::f(const vec3& wi, const vec3& wo, const vec3& n) const {
    if(bad_sample(wi, wo, n)) return zero<vec3>();

    float F = fresnel(wi, wo);
    vec3 BRDF = reflective_material->f(wi, wo, n);
    vec3 BTDF = transmissive_material->f(wi, wo, n);
	return F * BRDF + (1.0f - F) * BTDF;
}

WiSample DielectricBSDF::sample_wi(const vec3& wo, const vec3& n) const {
	WiSample r;

    if(randf() < 0.5) {
        r = reflective_material->sample_wi(wo, n);
        r.f *= fresnel(r.wi, wo);
    }
    else {
        r = transmissive_material->sample_wi(wo, n);
        r.f *= (1.0f - fresnel(r.wi, wo));
    }
    r.pdf *= 0.5;

	return r;
}

vec3 MetalBSDF::f(const vec3& wi, const vec3& wo, const vec3& n) const {
    if(bad_sample(wi, wo, n)) return zero<vec3>();

    float F = fresnel(wi, wo);
    vec3 BRDF = reflective_material->f(wi, wo, n);
	return F * BRDF * color;
}

WiSample MetalBSDF::sample_wi(const vec3& wo, const vec3& n) const {
    WiSample r = reflective_material->sample_wi(wo, n);
    r.f *= color;
	return r;
}


vec3 BSDFLinearBlend::f(const vec3& wi, const vec3& wo, const vec3& n) const {
    if(bad_sample(wi, wo, n)) return zero<vec3>();
    return w * bsdf0->f(wi, wo, n) + (1.0f - w) * bsdf1->f(wi, wo, n);
}

WiSample BSDFLinearBlend::sample_wi(const vec3& wo, const vec3& n) const {
    WiSample r;

    if(randf() < w) {
        r = bsdf0->sample_wi(wo, n);
    }
    else {
        r = bsdf1->sample_wi(wo, n);
    }

	return r;
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

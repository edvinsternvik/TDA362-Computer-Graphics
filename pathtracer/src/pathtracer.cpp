#include "pathtracer.hpp"
#include <algorithm>
#include "material.hpp"
#include "embree.hpp"
#include "sampling.hpp"
#include <omp.h>

using namespace std;
using namespace glm;

glm::mat3 tangent_space(glm::vec3 n) {
	float sign = copysignf(1.0f, n.z);
	const float a = -1.0f / (sign + n.z);
	const float b = n.x * n.y * a;
	glm::mat3 r;
	r[0] = glm::vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
	r[1] = glm::vec3(b, sign + n.y * n.y * a, -n.y);
	r[2] = n;
	return r;
}

namespace pathtracer {
///////////////////////////////////////////////////////////////////////////////
// Global variables
///////////////////////////////////////////////////////////////////////////////
Settings settings;
Environment environment;
Image rendered_image;
PointLight point_light;
std::vector<DiscLight> disc_lights;

///////////////////////////////////////////////////////////////////////////
// Restart rendering of image
///////////////////////////////////////////////////////////////////////////
void restart() {
	// No need to clear image,
	rendered_image.number_of_samples = 0;
}

int get_sample_count() {
	return std::max(rendered_image.number_of_samples - 1, 0);
}

///////////////////////////////////////////////////////////////////////////
// On window resize, window size is passed in, actual size of pathtraced
// image may be smaller (if we're subsampling for speed)
///////////////////////////////////////////////////////////////////////////
void resize(int w, int h) {
	rendered_image.width = w / settings.subsampling;
	rendered_image.height = h / settings.subsampling;
	rendered_image.data.resize(rendered_image.width * rendered_image.height);
	restart();
}

///////////////////////////////////////////////////////////////////////////
/// Return the radiance from a certain direction wi from the environment
/// map.
///////////////////////////////////////////////////////////////////////////
vec3 Lenvironment(const vec3& wi) {
	const float theta = acos(std::max(-1.0f, std::min(1.0f, wi.y)));
	float phi = atan(wi.z, wi.x);
	if(phi < 0.0f) phi = phi + 2.0f * M_PI;
	vec2 lookup = vec2(phi / (2.0 * M_PI), 1 - theta / M_PI);
	return environment.multiplier * environment.map.sample(lookup.x, lookup.y);
}

///////////////////////////////////////////////////////////////////////////
/// Calculate the radiance going from one point (r.hitPosition()) in one
/// direction (-r.d), through path tracing.
///////////////////////////////////////////////////////////////////////////
vec3 Li(Ray& primary_ray) {
	vec3 L = vec3(0.0f);
	vec3 path_throughput = vec3(1.0);
	Ray current_ray = primary_ray;

	///////////////////////////////////////////////////////////////////
	// Get the intersection information from the ray
	///////////////////////////////////////////////////////////////////
	Intersection hit = get_intersection(current_ray);
	///////////////////////////////////////////////////////////////////
	// Create a Material tree for evaluating brdfs and calculating
	// sample directions.
	///////////////////////////////////////////////////////////////////

	Diffuse diffuse(hit.material->m_data.m_color);
    MicrofacetBRDF microfacet(hit.material->m_data.m_roughness);
    DielectricBSDF dielectric(&microfacet, &diffuse, hit.material->m_data.m_fresnel);
    MetalBSDF metal(&microfacet, hit.material->m_data.m_color, hit.material->m_data.m_fresnel);
    BSDFLinearBlend metal_blend(hit.material->m_data.m_metalic, &metal, &dielectric);
    BSDF& mat = metal_blend;

    // Shoot shadow ray
    vec3 shadow_ray_origin = hit.position + EPSILON * hit.geometry_normal;
    vec3 shadow_ray_delta = point_light.position - shadow_ray_origin;
    Ray shadow_ray = Ray(
        shadow_ray_origin, normalize(shadow_ray_delta),
        0.0, length(shadow_ray_delta)
    );

	///////////////////////////////////////////////////////////////////
	// Calculate Direct Illumination from light.
	///////////////////////////////////////////////////////////////////
	if(!occluded(shadow_ray)) {
		const float distance_to_light = length(point_light.position - hit.position);
		const float falloff_factor = 1.0f / (distance_to_light * distance_to_light);
		vec3 Li = point_light.intensity_multiplier * point_light.color * falloff_factor;
		vec3 wi = normalize(point_light.position - hit.position);
		L = mat.f(wi, hit.wo, hit.shading_normal) * Li * std::max(0.0f, dot(wi, hit.shading_normal));
	}
	// Return the final outgoing radiance for the primary ray
	return L;
}

///////////////////////////////////////////////////////////////////////////
/// Used to homogenize points transformed with projection matrices
///////////////////////////////////////////////////////////////////////////
inline static glm::vec3 homogenize(const glm::vec4& p) {
	return glm::vec3(p * (1.f / p.w));
}

///////////////////////////////////////////////////////////////////////////
/// Trace one path per pixel and accumulate the result in an image
///////////////////////////////////////////////////////////////////////////
void trace_paths(const glm::mat4& V, const glm::mat4& P) {
	// Stop here if we have as many samples as we want
    if(rendered_image.number_of_samples > settings.max_paths_per_pixel
        && settings.max_paths_per_pixel != 0
    ) {
        return;
	}
	vec3 camera_pos = vec3(glm::inverse(V) * vec4(0.0f, 0.0f, 0.0f, 1.0f));
	// Trace one path per pixel (the omp parallel stuf magically distributes the
	// pathtracing on all cores of your CPU).
	int num_rays = 0;
	vector<vec4> local_image(rendered_image.width * rendered_image.height, vec4(0.0f));

#pragma omp parallel for
	for(int y = 0; y < rendered_image.height; y++) {
		for(int x = 0; x < rendered_image.width; x++) {
			vec3 color;
			Ray primary_ray;
			primary_ray.origin = camera_pos;
			// Create a ray that starts in the camera position and points toward
			// the current pixel on a virtual screen.
			vec2 screen_coords = vec2(
                (float(x) + randf()) / float(rendered_image.width),
			    (float(y) + randf()) / float(rendered_image.height)
            );
			// Calculate direction
			vec4 view_coords = vec4(
                screen_coords.x * 2.0f - 1.0f,
                screen_coords.y * 2.0f - 1.0f,
                1.0f, 1.0f
            );
			vec3 p = homogenize(inverse(P * V) * view_coords);
			primary_ray.direction = normalize(p - camera_pos);
			// Intersect ray with scene
			if(intersect(primary_ray)) {
				// If it hit something, evaluate the radiance from that point
				color = Li(primary_ray);
			}
			else {
				// Otherwise evaluate environment
				color = Lenvironment(primary_ray.direction);
			}
			// Accumulate the obtained radiance to the pixels color
			float n = float(rendered_image.number_of_samples);
			rendered_image.data[y * rendered_image.width + x] =
			    rendered_image.data[y * rendered_image.width + x] * (n / (n + 1.0f))
			    + (1.0f / (n + 1.0f)) * color;
		}
	}
	rendered_image.number_of_samples += 1;
}
}; // namespace pathtracer

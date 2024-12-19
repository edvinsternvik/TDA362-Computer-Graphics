#version 450

layout(location = 0) in vec2 in_uv;

layout(binding = 0) uniform SSAOUBO {
    mat4 inv_pv;
    mat4 projection_matrix;
    mat4 inv_projection_matrix;
};
layout(binding = 1) uniform sampler2D normal_sampler;
layout(binding = 2) uniform sampler2D depth_sampler;

layout(location = 0) out vec4 out_color;

#define PI 3.14159265359

vec3 homogenize(vec4 v) {
    return vec3((1.0 / v.w) * v);
}

vec3 perpendicular(vec3 v) {
    vec3 av = abs(v); 
    if (av.x < av.y)
        if (av.x < av.z) return vec3(0.0f, -v.z, v.y);
        else return vec3(-v.y, v.x, 0.0f);
    else
        if (av.y < av.z) return vec3(-v.z, 0.0f, v.x);
        else return vec3(-v.y, v.x, 0.0f);
}

// PCG random generator for 3 16-bit unsigned ints
uvec3 pcg3d16(uvec3 v)
{
	v = v * 12829u + 47989u;

	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;

	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;

	v ^= v >> 16u;
	return v;
}

// Conversion function to move from floats to uints, and back
vec3 pcg3d16f(vec3 v)
{
	uvec3 uv = floatBitsToUint(v);
	uv ^= uv >> 16u; // Make the info be contained in the lower 16 bits

	uvec3 m = pcg3d16(uv);

	return vec3(m & 0xFFFF) / vec3(0xFFFF);

	// Construct a float with half-open range [0,1) using low 23 bits.
	// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
	// From https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
	const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

	// Since the pcg3d16 function is only made to work for the lower 16 bits, we only use those
	// by shifting them to be the highest of the 23
	m <<= 7u;
    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    vec3 f = uintBitsToFloat(m);           // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

#define randf pcg3d16f

vec3 sampleHemisphereVolumeCosine(float idx)
{
	vec3 r = randf(vec3(gl_FragCoord.xy, idx));	
	vec3 ret;
	r.x *= 2 * PI;
	r.y = sqrt(r.y);
	r.y = min(r.y, 0.99);
	r.z = max(0.1, r.z);

	ret.x = r.y * cos(r.x);
	ret.y = r.y * sin(r.x);
	ret.z = sqrt(max(0, 1 - dot(ret.xy, ret.xy)));
	return ret * r.z;
}

void main() {
    float frag_depth = texture(depth_sampler, in_uv).r;
    vec4 ndc = vec4(
        in_uv.x * 2.0 - 1.0, in_uv.y * 2.0 - 1.0, 
        frag_depth, 1.0
    );
    vec3 vs_pos = homogenize(inv_projection_matrix * ndc);

    vec3 vs_normal = normalize(texture(normal_sampler, in_uv).xyz);
    vec3 vs_tangent = perpendicular(vs_normal);
    vec3 vs_bitangent = cross(vs_normal, vs_tangent);
    mat3 tbn = mat3(vs_tangent, vs_bitangent, vs_normal);

    float hemisphere_radius = 0.8;
    int nof_samples = 8;
    float visibility = 0.0;
    float bias = 0.1;
    for(int i = 0; i < nof_samples; i++) {
        // Project an hemishere sample onto the local base
        vec3 s = tbn * sampleHemisphereVolumeCosine(i);

        // compute view-space position of sample
        vec3 vs_sample_position = vs_pos + s * hemisphere_radius;

        // compute the ndc-coords of the sample
        vec3 sample_coords_ndc = homogenize(projection_matrix * vec4(vs_sample_position, 1.0));

        // Sample the depth-buffer at a texture coord based on the ndc-coord of the sample
        vec2 sample_coords_uv = sample_coords_ndc.xy * 0.5 + 0.5;
        float blocker_depth = texture(depth_sampler, sample_coords_uv).r;

        // Find the view-space coord of the blocker
        vec3 vs_blocker_pos = homogenize(inv_projection_matrix * 
            vec4(sample_coords_ndc.xy, blocker_depth, 1.0)
        );

        float range_check = smoothstep(0.0, 1.0, hemisphere_radius / abs(vs_blocker_pos.z - vs_sample_position.z));
        visibility += (vs_blocker_pos.z >= vs_sample_position.z + bias ? 1.0 : 0.0) * range_check;
    }

    visibility = 1.0 - visibility / float(nof_samples);

    out_color = vec4(vec3(visibility), 1.0);
}

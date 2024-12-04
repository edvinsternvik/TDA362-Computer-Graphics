#version 450

layout(binding = 0) uniform PostFXUBO {
    float time;
    int filter_size;
    bool enable_mushrooms;
    bool enable_mosaiac;
    bool enable_blur;
    bool enable_grayscale;
    bool enable_sepia;
};
layout(binding = 1) uniform sampler2D frame_sampler;

layout(location = 0) out vec4 out_color;

vec2 texture_coords(in sampler2D tex, vec2 pixel_coords) {
	return pixel_coords / textureSize(tex, 0);
}

vec3 to_sepia_tone(vec3 rgb_sample) {
	//-----------------------------------------------------------------
	// Variables used for YIQ/RGB color space conversion.
	//-----------------------------------------------------------------
	vec3 yiq_transform0 = vec3(0.299, 0.587, 0.144);
	vec3 yiq_transform1 = vec3(0.596, -0.275, -0.321);
	vec3 yiq_transform2 = vec3(0.212, -0.523, 0.311);

	vec3 yiq_inverse_transform0 = vec3(1, 0.956, 0.621);
	vec3 yiq_inverse_transform1 = vec3(1, -0.272, -0.647);
	vec3 yiq_inverse_transform2 = vec3(1, -1.105, 1.702);

	// transform to YIQ color space and set color information to sepia tone
	vec3 yiq = vec3(dot(yiq_transform0, rgb_sample), 0.2, 0.0);

	// inverse transform to RGB color space
	vec3 result = vec3(
        dot(yiq_inverse_transform0, yiq),
        dot(yiq_inverse_transform1, yiq),
        dot(yiq_inverse_transform2, yiq)
    );

	return result;
}

vec2 mushrooms(vec2 uv) {
	return uv + vec2(sin(time * 4.3127 + uv.y * 50.0) / 40.0, 0.0);
}

vec2 mosaiac(vec2 uv, vec2 texture_size, float pixels) {
    float aspect_ratio = texture_size.x / texture_size.y;
    return floor(uv * pixels / vec2(1.0, aspect_ratio))
        * vec2(1.0, aspect_ratio) / pixels;
}

vec3 blur(vec2 uv, vec2 texture_size) {
	vec3 result = vec3(0.0);
	float weight = 1.0 / (filter_size * filter_size);

	for(float i = -filter_size / 2; i <= filter_size / 2; i += 1.0) {
		for(float j = -filter_size / 2; j <= filter_size / 2; j += 1.0) {
            result += weight * texture(frame_sampler, uv + vec2(i, j) / texture_size).rgb;
		}
	}

	return result;
}


vec3 grayscale(vec3 rgb_sample) {
	return vec3(rgb_sample.r * 0.2126 + rgb_sample.g * 0.7152 + rgb_sample.b * 0.0722);
}

void main() {
    vec2 pixel_coords = gl_FragCoord.xy;
    vec2 texture_size = textureSize(frame_sampler, 0);
    vec2 uv = texture_coords(frame_sampler, pixel_coords);

    if(enable_mushrooms) uv = mushrooms(uv);
    if(enable_mosaiac) uv = mosaiac(uv, texture_size, 64.0);

    vec3 rgb_sample = texture(frame_sampler, uv).rgb;

    if(enable_blur) rgb_sample = blur(uv, texture_size);
    if(enable_grayscale) rgb_sample = grayscale(rgb_sample);
    if(enable_sepia) rgb_sample = to_sepia_tone(rgb_sample);

    out_color = vec4(rgb_sample, 1.0);
}

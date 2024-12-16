#version 420

layout(location = 0) in vec2 tex_coords;

layout(binding = 0) uniform sampler2D image;

layout(location = 0) out vec4 fragment_color;

void main() {
	fragment_color = texture(image, tex_coords);
}

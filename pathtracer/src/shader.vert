#version 420

layout(location = 0) in vec2 position;

layout(location = 0) out vec2 tex_coords;

void main() {
	gl_Position = vec4(position, 0.0, 1.0);
	tex_coords = 0.5 * (position + vec2(1, 1));
}

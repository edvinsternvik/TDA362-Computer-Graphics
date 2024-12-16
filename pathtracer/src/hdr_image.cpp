#include "hdr_image.hpp"

using namespace std;
using namespace glm;

void HDRImage::load(const string& filename) {
	stbi_set_flip_vertically_on_load(true);
	data = stbi_loadf(filename.c_str(), &width, &height, &components, 3);
	if(data == NULL) {
		exit(1);
	}
};

vec3 HDRImage::sample(float u, float v) {
	int x = int(u * width) % width;
	int y = int(v * height) % height;
    assert((y * width + x * 3 + 3) <= width * height * components);
	return vec3(
        data[(y * width + x) * 3 + 0],
        data[(y * width + x) * 3 + 1],
        data[(y * width + x) * 3 + 2]
    );
}

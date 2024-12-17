#include "embree.hpp"
#include <embree4/rtcore_buffer.h>
#include <embree4/rtcore_common.h>
#include <embree4/rtcore_device.h>
#include <embree4/rtcore_geometry.h>
#include <embree4/rtcore_ray.h>
#include <embree4/rtcore_scene.h>
#include <iostream>
#include <limits>
#include <map>

using namespace std;
using namespace glm;

namespace pathtracer {
///////////////////////////////////////////////////////////////////////////
// Global variables
///////////////////////////////////////////////////////////////////////////
RTCDevice embree_device = nullptr;
RTCScene embree_scene = nullptr;

///////////////////////////////////////////////////////////////////////////
// Build an acceleration structure for the scene
///////////////////////////////////////////////////////////////////////////
void build_bvh() {
	cout << "Embree building BVH..." << flush;
	rtcCommitScene(embree_scene);
	cout << "done.\n";
}

///////////////////////////////////////////////////////////////////////////
// Called when there is an embree error
///////////////////////////////////////////////////////////////////////////
void embree_error_handler(void* userval, const RTCError code, const char* str) {
	cout << "Embree ERROR: " << str << endl;
	exit(1);
}

///////////////////////////////////////////////////////////////////////////
// Used to map an Embree geometry ID to our scene Meshes and Materials
///////////////////////////////////////////////////////////////////////////
map<uint32_t, const Model*> map_geom_ID_to_model;
map<uint32_t, const Mesh*> map_geom_ID_to_mesh;

void init_embree() {
	///////////////////////////////////////////////////////////////////////
	// Lazy initialize embree on first use
	///////////////////////////////////////////////////////////////////////
	static bool embree_is_initialized = false;
	if(!embree_is_initialized) {
		cout << "Initializing embree..." << flush;
		embree_is_initialized = true;
		embree_device = rtcNewDevice(nullptr);
		rtcSetDeviceErrorFunction(embree_device, embree_error_handler, nullptr);
		cout << "done.\n";
	}
}

void reinit_scene() {
	init_embree();

	if(embree_scene) {
		rtcReleaseScene(embree_scene);
	}

	embree_scene = rtcNewScene(embree_device);
}

///////////////////////////////////////////////////////////////////////////
// Add a model to the embree scene
///////////////////////////////////////////////////////////////////////////
void add_model(const Model* model, const mat4& model_matrix) {
	///////////////////////////////////////////////////////////////////////
	// Lazy initialize embree on first use
	///////////////////////////////////////////////////////////////////////
	if(!embree_scene) {
		reinit_scene();
	}

	///////////////////////////////////////////////////////////////////////
	// Transform and add each mesh in the model as a geometry in embree,
	// and create mappings so that we can connect an embree geom_ID to a
	// Material.
	///////////////////////////////////////////////////////////////////////
	cout << "Adding " << model->m_name << " to embree scene..." << endl;
	for(auto& mesh : model->m_meshes) {
        RTCGeometry geom = rtcNewGeometry(embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);
		// Transform and commit vertices
        vec3* embree_vertices = (vec3*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), mesh.m_num_vertices);
		for(uint32_t i = 0; i < mesh.m_num_vertices; i++) {
			embree_vertices[i] = vec3(model_matrix * vec4(model->m_positions[mesh.m_start_index + i], 1.0f));
		}
        unsigned int* embree_tri_idxs = (unsigned int*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned int), mesh.m_num_vertices / 3);
		for(uint32_t i = 0; i < mesh.m_num_vertices; i++) {
			embree_tri_idxs[i] = i;
		}

        rtcCommitGeometry(geom);

        uint32_t geom_ID = rtcAttachGeometry(embree_scene, geom);
		map_geom_ID_to_mesh[geom_ID] = &mesh;
		map_geom_ID_to_model[geom_ID] = model;

        rtcReleaseGeometry(geom);
	}
    rtcCommitScene(embree_scene);
	cout << "done.\n";
}

///////////////////////////////////////////////////////////////////////////
// Extract an intersection from an embree ray.
///////////////////////////////////////////////////////////////////////////
Intersection get_intersection(const Ray& r) {
    assert(map_geom_ID_to_model.contains(r.geomID));
    assert(map_geom_ID_to_mesh.contains(r.geomID));
	const Model* model = map_geom_ID_to_model[r.geomID];
	const Mesh* mesh = map_geom_ID_to_mesh[r.geomID];

	Intersection i;
	i.material = &(model->m_materials[mesh->m_material_index]);
    assert((((mesh->m_start_index / 3) + r.primID) * 3 + 3) <= model->m_normals.size());
	vec3 n0 = model->m_normals[((mesh->m_start_index / 3) + r.primID) * 3 + 0];
	vec3 n1 = model->m_normals[((mesh->m_start_index / 3) + r.primID) * 3 + 1];
	vec3 n2 = model->m_normals[((mesh->m_start_index / 3) + r.primID) * 3 + 2];
	float w = 1.0f - (r.u + r.v);
	i.shading_normal = normalize(w * n0 + r.u * n1 + r.v * n2);
	i.geometry_normal = normalize(r.n);
	i.position = r.o + r.tfar * r.d;
	i.wo = normalize(-r.d);

    assert((((mesh->m_start_index / 3) + r.primID) * 3 + 3) <= model->m_texture_coordinates.size());
	vec2 uv0 = model->m_texture_coordinates[((mesh->m_start_index / 3) + r.primID) * 3 + 0];
	vec2 uv1 = model->m_texture_coordinates[((mesh->m_start_index / 3) + r.primID) * 3 + 1];
	vec2 uv2 = model->m_texture_coordinates[((mesh->m_start_index / 3) + r.primID) * 3 + 2];
	i.uv = w * uv0 + r.u * uv1 + r.v * uv2;
	return i;
}

///////////////////////////////////////////////////////////////////////////
// Test a ray against the scene and find the closest intersection
///////////////////////////////////////////////////////////////////////////
bool intersect(Ray& r) {
    struct RTCRayHit rayhit = {};
    rayhit.ray.org_x = r.o.x;
    rayhit.ray.org_y = r.o.y;
    rayhit.ray.org_z = r.o.z;
    rayhit.ray.tnear = r.tnear;
    rayhit.ray.dir_x = r.d.x;
    rayhit.ray.dir_y = r.d.y;
    rayhit.ray.dir_z = r.d.z;
    rayhit.ray.time = r.time;
    rayhit.ray.tfar = r.tfar;
    rayhit.ray.mask = r.mask;
    rayhit.ray.flags = 0;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(embree_scene, &rayhit);
    r.o.x = rayhit.ray.org_x;
    r.o.y = rayhit.ray.org_y;
    r.o.z = rayhit.ray.org_z;
    r.d.x = rayhit.ray.dir_x;
    r.d.y = rayhit.ray.dir_y;
    r.d.z = rayhit.ray.dir_z;
    r.tnear = rayhit.ray.tnear;
    r.tfar = rayhit.ray.tfar;
    r.time = rayhit.ray.time;
    r.mask = rayhit.ray.mask;
    r.n.x = rayhit.hit.Ng_x;
    r.n.y = rayhit.hit.Ng_y;
    r.n.z = rayhit.hit.Ng_z;
    r.u = rayhit.hit.u;
    r.v = rayhit.hit.v;
    r.geomID = rayhit.hit.geomID;
    r.primID = rayhit.hit.primID;
    r.instID = rayhit.hit.instID[0];
	return rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID;
}

///////////////////////////////////////////////////////////////////////////
// Test whether a ray is intersected by the scene (do not return an
// intersection).
///////////////////////////////////////////////////////////////////////////
bool occluded(Ray& r) {
    struct RTCRay ray = {};
    ray.org_x = r.o.x;
    ray.org_y = r.o.y;
    ray.org_z = r.o.z;
    ray.tnear = r.tnear;
    ray.dir_x = r.d.x;
    ray.dir_y = r.d.y;
    ray.dir_z = r.d.z;
    ray.time = r.time;
    ray.tfar = r.tfar;
    ray.mask = r.mask;
    ray.time = r.time;
    ray.flags = 0;

    rtcOccluded1(embree_scene, &ray);
    r.o.x = ray.org_x;
    r.o.y = ray.org_y;
    r.o.z = ray.org_z;
    r.d.x = ray.dir_x;
    r.d.y = ray.dir_y;
    r.d.z = ray.dir_z;
    r.tnear = ray.tnear;
    r.tfar = ray.tfar;
    r.time = ray.time;
    r.mask = ray.mask;

    return ray.tfar == -std::numeric_limits<float>::infinity();
}
} // namespace pathtracer

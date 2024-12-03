#include "model.hpp"
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <tiny_obj_loader.h>
#include <filesystem>
#include <vulkan/vulkan_core.h>

void Material::destroy(VkDevice device) {
    if(m_color_texture.has_value()) m_color_texture->destroy(device);
    if(m_metalic_texture.has_value()) m_metalic_texture->destroy(device);
    if(m_fresnel_texture.has_value()) m_fresnel_texture->destroy(device);
    if(m_roughness_texture.has_value()) m_roughness_texture->destroy(device);
    if(m_emission_texture.has_value()) m_emission_texture->destroy(device);
}

Model load_model_from_file(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    const char* file_name
) {
    std::filesystem::path file_path(file_name);
    std::filesystem::path base_path = file_path.parent_path();

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;
	// Expect '.mtl' file in the same directory and triangulate meshes
	bool ret = tinyobj::LoadObj(
        &attrib, &shapes, &materials, &warn, &err,
	    file_path.c_str(), base_path.c_str()
    );
	if(!err.empty() || !warn.empty()) {
        throw std::runtime_error(warn + err);
	}
	if(!ret) {
		exit(1);
	}
	Model model = {};
	model.m_name = std::string(file_name);

    auto get_path = [&](const std::string& in_path) {
		std::string out_path;
		for(const char c : in_path) {
			if(c == '\\') {
				if(out_path.back() != '/') {
					out_path += '/';
				}
			}
			else {
				out_path += c;
			}
		}
		return base_path / out_path;
    };

    // Materials
	for(const auto& m : materials) {
		Material material;
		material.m_name = m.name;
		material.m_data.m_color = glm::vec3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
		if(m.diffuse_texname != "") {
            material.m_color_texture = load_texture_from_image(
                device, physical_device,
                command_pool, command_queue,
                get_path(m.diffuse_texname).c_str()
            );
		}
		material.m_data.m_metalic = m.metallic;
		if(m.metallic_texname != "") {
            material.m_metalic_texture = load_texture_from_image(
                device, physical_device,
                command_pool, command_queue,
                get_path(m.metallic_texname).c_str()
            );
		}
		material.m_data.m_fresnel = m.specular[0];
		if(m.specular_texname != "") {
            material.m_fresnel_texture = load_texture_from_image(
                device, physical_device,
                command_pool, command_queue,
                get_path(m.specular_texname).c_str()
            );
		}
		material.m_data.m_roughness = m.roughness;
		if(m.roughness_texname != "") {
            material.m_roughness_texture = load_texture_from_image(
                device, physical_device,
                command_pool, command_queue,
                get_path(m.roughness_texname).c_str()
            );
		}
		material.m_data.m_emission = glm::vec3(m.emission[0], m.emission[1], m.emission[2]);
		if(m.emissive_texname != "") {
            material.m_emission_texture = load_texture_from_image(
                device, physical_device,
                command_pool, command_queue,
                get_path(m.emissive_texname).c_str()
            );
		}
		material.m_data.m_transparency = m.transmittance[0];
		material.m_data.m_ior = m.ior;
		model.m_materials.push_back(material);
	}

	uint64_t number_of_vertices = 0;
	for(const auto& shape : shapes) {
		number_of_vertices += shape.mesh.indices.size();
	}

    struct Vertex {
        float x, y, z;
        float nx, ny, nz;
        float u, v;
    };

    std::vector<Vertex> vertices(number_of_vertices, Vertex{});

    // Generate normals
	std::vector<glm::vec4> auto_normals(attrib.vertices.size() / 3);
	for(const auto& shape : shapes) {
		for(int face = 0; face < int(shape.mesh.indices.size()) / 3; face++) {
			glm::vec3 v0 = glm::vec3(
                attrib.vertices[shape.mesh.indices[face * 3 + 0].vertex_index * 3 + 0],
			    attrib.vertices[shape.mesh.indices[face * 3 + 0].vertex_index * 3 + 1],
			    attrib.vertices[shape.mesh.indices[face * 3 + 0].vertex_index * 3 + 2]
            );
			glm::vec3 v1 = glm::vec3(
                attrib.vertices[shape.mesh.indices[face * 3 + 1].vertex_index * 3 + 0],
			    attrib.vertices[shape.mesh.indices[face * 3 + 1].vertex_index * 3 + 1],
			    attrib.vertices[shape.mesh.indices[face * 3 + 1].vertex_index * 3 + 2]
            );
			glm::vec3 v2 = glm::vec3(
                attrib.vertices[shape.mesh.indices[face * 3 + 2].vertex_index * 3 + 0],
			    attrib.vertices[shape.mesh.indices[face * 3 + 2].vertex_index * 3 + 1],
			    attrib.vertices[shape.mesh.indices[face * 3 + 2].vertex_index * 3 + 2]
            );

			glm::vec3 e0 = glm::normalize(v1 - v0);
			glm::vec3 e1 = glm::normalize(v2 - v0);
			glm::vec3 face_normal = cross(e0, e1);

			auto_normals[shape.mesh.indices[face * 3 + 0].vertex_index] += glm::vec4(face_normal, 1.0f);
			auto_normals[shape.mesh.indices[face * 3 + 1].vertex_index] += glm::vec4(face_normal, 1.0f);
			auto_normals[shape.mesh.indices[face * 3 + 2].vertex_index] += glm::vec4(face_normal, 1.0f);
		}
	}
	for(auto& normal : auto_normals) {
		normal = (1.0f / normal.w) * normal;
	}
    // Generate meshes
	int vertices_so_far = 0;
	for(int s = 0; s < shapes.size(); ++s) {
		const auto& shape = shapes[s];
		int next_material_index = shape.mesh.material_ids[0];
		int next_material_starting_face = 0;
		std::vector<bool> finished_materials(materials.size(), false);
		int number_of_materials_in_shape = 0;
		while(next_material_index != -1) {
			int current_material_index = next_material_index;
			int current_material_starting_face = next_material_starting_face;
			next_material_index = -1;
			next_material_starting_face = -1;

			// Process a new Mesh with a unique material
			Mesh mesh;
			mesh.m_name = shape.name + "_" + materials[current_material_index].name;
			mesh.m_material_index = current_material_index;
			mesh.m_start_index = vertices_so_far;
			number_of_materials_in_shape += 1;

			uint64_t number_of_faces = shape.mesh.indices.size() / 3;
			for(int i = current_material_starting_face; i < number_of_faces; i++) {
				if(shape.mesh.material_ids[i] != current_material_index) {
					if(next_material_index >= 0)
						continue;
					else if(finished_materials[shape.mesh.material_ids[i]])
						continue;
					else { // Found a new material that we have not processed.
						next_material_index = shape.mesh.material_ids[i];
						next_material_starting_face = i;
					}
				}
				else {
                    // Generate vertices
					for(int j = 0; j < 3; j++) {
                        Vertex& vertex = vertices[vertices_so_far + j];

                        vertex.x = attrib.vertices[shape.mesh.indices[i * 3 + j].vertex_index * 3 + 0];
                        vertex.y = attrib.vertices[shape.mesh.indices[i * 3 + j].vertex_index * 3 + 1];
                        vertex.z = attrib.vertices[shape.mesh.indices[i * 3 + j].vertex_index * 3 + 2];

						if(shape.mesh.indices[i * 3 + j].normal_index == -1) {
							vertex.nx = auto_normals[shape.mesh.indices[i * 3 + j].vertex_index].x;
							vertex.ny = auto_normals[shape.mesh.indices[i * 3 + j].vertex_index].y;
							vertex.nz = auto_normals[shape.mesh.indices[i * 3 + j].vertex_index].z;
						}
						else {
                            vertex.nx = attrib.normals[shape.mesh.indices[i * 3 + j].normal_index * 3 + 0];
                            vertex.ny = attrib.normals[shape.mesh.indices[i * 3 + j].normal_index * 3 + 1];
                            vertex.nz = attrib.normals[shape.mesh.indices[i * 3 + j].normal_index * 3 + 2];
						}
						if(shape.mesh.indices[i * 3 + j].texcoord_index == -1) {
							vertex.u = 0.0;
							vertex.v = 0.0;
						}
						else {
                            vertex.u = attrib.texcoords[shape.mesh.indices[i * 3 + j].texcoord_index * 2 + 0];
                            vertex.v = attrib.texcoords[shape.mesh.indices[i * 3 + j].texcoord_index * 2 + 1];
						}
					}
					vertices_so_far += 3;
				}
			}

			mesh.m_num_vertices = vertices_so_far - mesh.m_start_index;
			model.m_meshes.push_back(mesh);
			finished_materials[current_material_index] = true;
		}
		if(number_of_materials_in_shape == 1) {
			model.m_meshes.back().m_name = shape.name;
		}
	}

	std::sort(
        model.m_meshes.begin(), model.m_meshes.end(),
        [](const Mesh& a, const Mesh& b) { return a.m_name < b.m_name; }
    );

    model.m_vertex_buffer = create_buffer(
        device,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        sizeof(Vertex) * vertices.size()
    );

    model.m_vertex_buffer_memory = allocate_buffer_memory(
        device, physical_device,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        model.m_vertex_buffer
    );

    write_buffer_staged(
        device, physical_device,
        command_queue, command_pool,
        model.m_vertex_buffer,
        vertices.data(), sizeof(Vertex) * vertices.size()
    );

    return model;
}

void Model::destroy(VkDevice device) {
    for(auto m : m_materials) m.destroy(device);
    vkDestroyBuffer(device, m_vertex_buffer, nullptr);
    vkFreeMemory(device, m_vertex_buffer_memory, nullptr);
}

FrameData create_frame_data(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    VkDescriptorSetLayout descriptor_set_layout,
    const size_t max_objects
) {
    FrameData frame_data;
    frame_data.m_max_objects = max_objects;
    frame_data.m_mvp_uniform_buffers = std::vector<VkBuffer>(max_objects);
    frame_data.m_mvp_uniform_memory = std::vector<VkDeviceMemory>(max_objects);
    frame_data.m_material_uniform_buffers = std::vector<VkBuffer>(max_objects);
    frame_data.m_material_uniform_memory = std::vector<VkDeviceMemory>(max_objects);
    frame_data.m_descriptor_sets = std::vector<VkDescriptorSet>(max_objects);

    std::array<VkDescriptorPoolSize, 7> descriptor_pool_sizes = {};
    descriptor_pool_sizes[0].descriptorCount = max_objects;
    descriptor_pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_pool_sizes[1].descriptorCount = max_objects;
    descriptor_pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    for(size_t i = 2; i < 7; ++i) {
        descriptor_pool_sizes[i].descriptorCount = max_objects;
        descriptor_pool_sizes[i].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
    descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.maxSets = max_objects;
    descriptor_pool_create_info.poolSizeCount = descriptor_pool_sizes.size();
    descriptor_pool_create_info.pPoolSizes = descriptor_pool_sizes.data();
    vkCreateDescriptorPool(device, &descriptor_pool_create_info, nullptr, &frame_data.m_descriptor_pool);

    for(size_t i = 0; i < max_objects; ++i) {
        frame_data.m_mvp_uniform_buffers[i] = create_buffer(
            device,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            3 * sizeof(glm::mat4)
        );
        frame_data.m_mvp_uniform_memory[i] = allocate_buffer_memory(
            device, physical_device,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            frame_data.m_mvp_uniform_buffers[i]
        );

        frame_data.m_material_uniform_buffers[i] = create_buffer(
            device,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            sizeof(MaterialData)
        );
        frame_data.m_material_uniform_memory[i] = allocate_buffer_memory(
            device, physical_device,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            frame_data.m_material_uniform_buffers[i]
        );
    }

    std::vector<VkDescriptorSetLayout> descriptor_set_layouts(max_objects, descriptor_set_layout);
    VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
    descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_set_allocate_info.descriptorPool = frame_data.m_descriptor_pool;
    descriptor_set_allocate_info.descriptorSetCount = max_objects;
    descriptor_set_allocate_info.pSetLayouts = descriptor_set_layouts.data();
    vkAllocateDescriptorSets(device, &descriptor_set_allocate_info, frame_data.m_descriptor_sets.data());

    // Create empty image
    std::array<uint8_t, 4> empty_image_data = { 0, 0, 0, 0 };
    frame_data.m_empty_image = create_image(
        device,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_TILING_OPTIMAL,
        1, 1, 1
    );
    frame_data.m_empty_image_memory = allocate_image_memory(
        device, physical_device,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        frame_data.m_empty_image
    );
    transition_image_layout(
        device, command_pool, command_queue,
        frame_data.m_empty_image,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1
    );
    write_image_staged(
        device, physical_device, command_queue, command_pool,
        frame_data.m_empty_image,
        empty_image_data.data(), empty_image_data.size(),
        1, 1
    );
    transition_image_layout(
        device, command_pool, command_queue,
        frame_data.m_empty_image,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        1
    );

    {
        VkImageViewCreateInfo view_create_info = {};
        view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_create_info.image = frame_data.m_empty_image;
        view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_create_info.format = VK_FORMAT_R8G8B8A8_SRGB;
        view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_create_info.subresourceRange.baseMipLevel = 0;
        view_create_info.subresourceRange.levelCount = 1;
        view_create_info.subresourceRange.baseArrayLayer = 0;
        view_create_info.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &view_create_info, nullptr, &frame_data.m_empty_image_view);
    }

    return frame_data;
}

void update_frame_data(
    VkDevice device,
    FrameData* frame_data,
    VkSampler sampler,
    const std::vector<Object*>& objects,
    const std::vector<Model>& models,
    glm::mat4 view_matrix,
    glm::mat4 projection_matrix
) {
    size_t descriptor_index = 0;
    for(size_t i = 0; i < objects.size(); ++i) {
        const Model& model = models[objects[i]->m_model_index];

        glm::mat4 model_matrix = glm::identity<glm::mat4>()
            * glm::translate(glm::identity<glm::mat4>(), objects[i]->position)
            * glm::mat4_cast(objects[i]->orientation)
            * glm::scale(glm::identity<glm::mat4>(), objects[i]->scale);

        glm::mat4 model_view_matrix =
            view_matrix
            * model_matrix;

        glm::mat4 model_view_projection_matrix =
            projection_matrix
            * model_view_matrix;

        for(size_t j = 0; j < model.m_meshes.size(); ++j) {
            const Mesh& mesh = model.m_meshes[j];
            const Material& material = model.m_materials[mesh.m_material_index];

            VkImageView color_view = frame_data->m_empty_image_view;
            if(material.m_color_texture.has_value()) {
                color_view = material.m_color_texture->m_image_view;
            }
            VkImageView metallic_view = frame_data->m_empty_image_view;
            if(material.m_metalic_texture.has_value()) {
                metallic_view = material.m_metalic_texture->m_image_view;
            }
            VkImageView fresnel_view = frame_data->m_empty_image_view;
            if(material.m_fresnel_texture.has_value()) {
                fresnel_view = material.m_fresnel_texture->m_image_view;
            }
            VkImageView roughness_view = frame_data->m_empty_image_view;
            if(material.m_roughness_texture.has_value()) {
                roughness_view = material.m_roughness_texture->m_image_view;
            }
            VkImageView emission_view = frame_data->m_empty_image_view;
            if(material.m_emission_texture.has_value()) {
                emission_view = material.m_emission_texture->m_image_view;
            }

            VkDescriptorBufferInfo descriptor_mvp_buffer_info = {};
            descriptor_mvp_buffer_info.buffer = frame_data->m_mvp_uniform_buffers[descriptor_index];
            descriptor_mvp_buffer_info.range = 3 * sizeof(glm::mat4);
            descriptor_mvp_buffer_info.offset = 0;

            VkDescriptorBufferInfo descriptor_material_buffer_info = {};
            descriptor_material_buffer_info.buffer = frame_data->m_material_uniform_buffers[descriptor_index];
            descriptor_material_buffer_info.range = sizeof(MaterialData);
            descriptor_material_buffer_info.offset = 0;

            std::array<VkDescriptorImageInfo, 5> descriptor_image_infos = {};
            for(size_t i = 0; i < 5; ++i) {
                descriptor_image_infos[i].sampler = sampler;
                descriptor_image_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            descriptor_image_infos[0].imageView = color_view;
            descriptor_image_infos[1].imageView = metallic_view;
            descriptor_image_infos[2].imageView = fresnel_view;
            descriptor_image_infos[3].imageView = roughness_view;
            descriptor_image_infos[4].imageView = emission_view;

            std::array<VkWriteDescriptorSet, 7> descriptor_set_writes = {};
            descriptor_set_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_set_writes[0].descriptorCount = 1;
            descriptor_set_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_set_writes[0].dstSet = frame_data->m_descriptor_sets[descriptor_index];
            descriptor_set_writes[0].dstBinding = 0;
            descriptor_set_writes[0].dstArrayElement = 0;
            descriptor_set_writes[0].pBufferInfo = &descriptor_mvp_buffer_info;
            descriptor_set_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_set_writes[1].descriptorCount = 1;
            descriptor_set_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_set_writes[1].dstSet = frame_data->m_descriptor_sets[descriptor_index];
            descriptor_set_writes[1].dstBinding = 1;
            descriptor_set_writes[1].dstArrayElement = 0;
            descriptor_set_writes[1].pBufferInfo = &descriptor_material_buffer_info;
            for(size_t i = 0; i < 5; ++i) {
                descriptor_set_writes[2 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptor_set_writes[2 + i].descriptorCount = 1;
                descriptor_set_writes[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptor_set_writes[2 + i].dstSet = frame_data->m_descriptor_sets[descriptor_index];
                descriptor_set_writes[2 + i].dstBinding = 2 + i;
                descriptor_set_writes[2 + i].dstArrayElement = 0;
                descriptor_set_writes[2 + i].pImageInfo = &descriptor_image_infos[i];
            }
            vkUpdateDescriptorSets(
                device,
                descriptor_set_writes.size(),
                descriptor_set_writes.data(),
                0, nullptr
            );

            std::array<glm::mat4, 2>* mvp_mapping;
            vkMapMemory(
                device,
                frame_data->m_mvp_uniform_memory[descriptor_index],
                0,
                3 * sizeof(glm::mat4),
                0,
                (void**)&mvp_mapping
            );

            (*(std::array<glm::mat4, 3>*)(mvp_mapping))[0] = model_matrix;
            (*(std::array<glm::mat4, 3>*)(mvp_mapping))[1] = model_view_matrix;
            (*(std::array<glm::mat4, 3>*)(mvp_mapping))[2] = model_view_projection_matrix;

            vkUnmapMemory(device, frame_data->m_mvp_uniform_memory[descriptor_index]);

            MaterialData* material_mapping;
            vkMapMemory(
                device,
                frame_data->m_material_uniform_memory[descriptor_index],
                0,
                sizeof(MaterialData),
                0,
                (void**)&material_mapping
            );

            *(MaterialData*)(material_mapping) = material.m_data;

            vkUnmapMemory(device, frame_data->m_material_uniform_memory[descriptor_index]);

            descriptor_index++;
        }
    }
}

void destroy_frame_data(
    VkDevice device,
    FrameData& frame_data
) {
    vkDestroyDescriptorPool(device, frame_data.m_descriptor_pool, nullptr);
    for(auto ub : frame_data.m_mvp_uniform_buffers) {
        vkDestroyBuffer(device, ub, nullptr);
    }
    for(auto um : frame_data.m_mvp_uniform_memory) {
        vkFreeMemory(device, um, nullptr);
    }
    for(auto ub : frame_data.m_material_uniform_buffers) {
        vkDestroyBuffer(device, ub, nullptr);
    }
    for(auto um : frame_data.m_material_uniform_memory) {
        vkFreeMemory(device, um, nullptr);
    }
    vkDestroyImage(device, frame_data.m_empty_image, nullptr);
    vkFreeMemory(device, frame_data.m_empty_image_memory, nullptr);
    vkDestroyImageView(device, frame_data.m_empty_image_view, nullptr);
}

VkDescriptorSetLayout create_model_descriptor_set_layout(
    VkDevice device
) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(7);
    bindings[0].binding = 0;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].pImmutableSamplers = nullptr;

    bindings[1].binding = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].pImmutableSamplers = nullptr;

    for(size_t i = 0; i < 5; ++i) {
        bindings[2 + i].binding = 2 + i;
        bindings[2 + i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[2 + i].descriptorCount = 1;
        bindings[2 + i].pImmutableSamplers = nullptr;
    }

    return create_descriptor_set_layout(
        device,
        bindings
    );
}

std::pair<
    VkVertexInputBindingDescription,
    std::vector<VkVertexInputAttributeDescription>
> create_model_attributes() {
    VkVertexInputBindingDescription binding_description = {};
    binding_description.binding = 0;
    binding_description.stride = 8 * sizeof(float);
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription position_attribute = {};
    position_attribute.binding = 0;
    position_attribute.location = 0;
    position_attribute.offset = 0;
    position_attribute.format = VK_FORMAT_R32G32B32_SFLOAT;

    VkVertexInputAttributeDescription normal_attribute = {};
    normal_attribute.binding = 0;
    normal_attribute.location = 1;
    normal_attribute.offset = 3 * sizeof(float);
    normal_attribute.format = VK_FORMAT_R32G32B32_SFLOAT;

    VkVertexInputAttributeDescription uv_attribute = {};
    uv_attribute.binding = 0;
    uv_attribute.location = 2;
    uv_attribute.offset = 6 * sizeof(float);
    uv_attribute.format = VK_FORMAT_R32G32_SFLOAT;

    return {
        binding_description,
        { position_attribute, normal_attribute, uv_attribute }
    };
}

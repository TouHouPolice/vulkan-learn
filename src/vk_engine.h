// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <deque>
#include <functional>
#include <vector>

#include <vk_types.h>
#include <glm/glm.hpp>

#include "vk_mesh.h"

struct Material
{
	VkDescriptorSet textureSet{ VK_NULL_HANDLE }; //texture defaulted to null
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct RenderObject {
	Mesh* mesh;

	Material* material;

	glm::mat4 transformMatrix;
};

struct MeshPushConstants
{
	glm::vec4 data;
	glm::mat4 render_matrix;
};

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function)
	{
				deletors.emplace_back(function);
	}

	void flush()
	{
		while (!deletors.empty())
		{
			deletors.front()();
			deletors.pop_front();
		}
	}
};

struct GPUCameraData
{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewProj;
};

struct GPUSceneData {
	glm::vec4 fogColor; // w is for exponent
	glm::vec4 fogDistances; //x for min, y for max, zw unused.
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; //w for sun power
	glm::vec4 sunlightColor;
};

struct GPUObjectData {
	glm::mat4 modelMatrix;
};

struct FrameData
{
	VkFence _renderFence;

	VkSemaphore _presentSemaphore;
	VkSemaphore _renderSemaphore;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	//buffer that holds a single GPUCameraData to use when rendering
	AllocatedBuffer _cameraBuffer;
	VkDescriptorSet _globalDescriptorSet;

	AllocatedBuffer _objectBuffer;
	VkDescriptorSet _objectDescriptor;
};

struct UploadContext {
	VkFence _uploadFence;
	VkCommandPool _commandPool;
	VkCommandBuffer _commandBuffer;
};

struct Texture
{
	AllocatedImage image;
	VkImageView imageView;
};


constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber {0};

	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };


	VkInstance _instance; // Vulkan library handle
	VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
	VkPhysicalDevice _chosenGPU; // GPU chosen as the default device
	VkDevice _device; // Vulkan device for commands
	VkSurfaceKHR _surface; // Vulkan window surface

	VkSwapchainKHR _swapchain; // from other articles

	// image format expected by the windowing system
	VkFormat _swapchainImageFormat;

	//array of images from the swapchain
	std::vector<VkImage> _swapchainImages;

	//array of image-views from the swapchain
	std::vector<VkImageView> _swapchainImageViews;

	VkQueue _graphicsQueue; //queue we will submit to
	uint32_t _graphicsQueueFamily; //family of that queue

	VkRenderPass _renderPass;
	std::vector<VkFramebuffer> _framebuffers;

	UploadContext _uploadContext;

	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	std::unordered_map<std::string, Texture> _loadedTextures;

private:
	FrameData _frames[FRAME_OVERLAP];

	int _selectedShader{ 0 };
	

	VkPipeline _meshPipeline;
	VkPipelineLayout _meshPipelineLayout;
	VkPipelineLayout texturedPipeLayout;

	AllocatedImage _depthImage;
	VkImageView _depthImageView;
	VkFormat _depthFormat;

	std::vector<RenderObject> _renderables;

	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Mesh> _meshes;

	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorSetLayout _objectSetLayout;
	VkDescriptorSetLayout _singleTextureSetLayout;

	VkDescriptorPool _descriptorPool;

	VkPhysicalDeviceProperties _gpuProperties;

	GPUSceneData _sceneParameters;
	AllocatedBuffer _sceneParameterBuffer;

	
private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_default_renderpass();
	void init_framebuffers();
	void init_sync_structures();
	bool load_shader_module(const char* filepath, VkShaderModule* outShaderModule);
	void init_pipelines();
	void init_scene();

	void load_meshes();
	void upload_mesh(Mesh& mesh);

	//create material and add it to the map
	Material* create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);

	//returns nullptr if it can't be found
	Material* get_material(const std::string& name);

	//returns nullptr if it can't be found
	Mesh* get_mesh(const std::string& name);

	//our draw function
	void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);

	FrameData& get_current_frame();

	
	void init_descriptors();

	size_t pad_uniform_buffer_size(size_t originalSize);

	
	
public:
	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	void load_images();
};

class PipelineBuilder
{
public:
	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	VkPipelineColorBlendAttachmentState _colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineLayout _pipelineLayout;

	VkPipeline build_pipeline(VkDevice device, VkRenderPass renderPass);

public:
	VkPipelineDepthStencilStateCreateInfo _depthStencil;

};




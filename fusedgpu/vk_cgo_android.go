//go:build birdkit_native_vk && android && cgo

package fusedgpu

/*
#cgo LDFLAGS: -ldl -llog
#include <dlfcn.h>
#include <android/log.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define VK_LOG(...) __android_log_print(ANDROID_LOG_INFO, "birdkit-vk", __VA_ARGS__)

static void bk_ALog(const char *msg) {
	VK_LOG("%s", msg ? msg : "");
}

typedef struct {
	void *lib;
	PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
	PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;
	PFN_vkCreateInstance vkCreateInstance;
	PFN_vkDestroyInstance vkDestroyInstance;
	PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
	PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
	PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
	PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
	PFN_vkCreateDevice vkCreateDevice;
	PFN_vkDestroyDevice vkDestroyDevice;
	PFN_vkGetDeviceQueue vkGetDeviceQueue;
	PFN_vkCreateCommandPool vkCreateCommandPool;
	PFN_vkDestroyCommandPool vkDestroyCommandPool;
	PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
	PFN_vkFreeCommandBuffers vkFreeCommandBuffers;
	PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
	PFN_vkEndCommandBuffer vkEndCommandBuffer;
	PFN_vkCmdBindPipeline vkCmdBindPipeline;
	PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
	PFN_vkCmdDispatch vkCmdDispatch;
	PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
	PFN_vkQueueSubmit vkQueueSubmit;
	PFN_vkQueueWaitIdle vkQueueWaitIdle;
	PFN_vkDeviceWaitIdle vkDeviceWaitIdle;
	PFN_vkCreateBuffer vkCreateBuffer;
	PFN_vkDestroyBuffer vkDestroyBuffer;
	PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
	PFN_vkAllocateMemory vkAllocateMemory;
	PFN_vkFreeMemory vkFreeMemory;
	PFN_vkBindBufferMemory vkBindBufferMemory;
	PFN_vkMapMemory vkMapMemory;
	PFN_vkUnmapMemory vkUnmapMemory;
	PFN_vkCreateShaderModule vkCreateShaderModule;
	PFN_vkDestroyShaderModule vkDestroyShaderModule;
	PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout;
	PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout;
	PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
	PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;
	PFN_vkCreateComputePipelines vkCreateComputePipelines;
	PFN_vkDestroyPipeline vkDestroyPipeline;
	PFN_vkCreateDescriptorPool vkCreateDescriptorPool;
	PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool;
	PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets;
	PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets;
	PFN_vkResetCommandPool vkResetCommandPool;
	PFN_vkCreateFence vkCreateFence;
	PFN_vkDestroyFence vkDestroyFence;
	PFN_vkWaitForFences vkWaitForFences;
	PFN_vkResetFences vkResetFences;
} vkLoader;

static void *bk_vkSym(vkLoader *L, VkInstance inst, VkDevice dev, const char *name) {
	void *p = NULL;
	if (dev && L->vkGetDeviceProcAddr)
		p = (void *)L->vkGetDeviceProcAddr(dev, name);
	if (!p && L->vkGetInstanceProcAddr)
		p = (void *)L->vkGetInstanceProcAddr(inst, name);
	if (!p && L->lib)
		p = dlsym(L->lib, name);
	return p;
}

static int bk_vkInitLoader(vkLoader *L) {
	memset(L, 0, sizeof(*L));
	L->lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
	if (!L->lib) {
		VK_LOG("dlopen libvulkan.so failed: %s", dlerror());
		return -1;
	}
	L->vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(L->lib, "vkGetInstanceProcAddr");
	if (!L->vkGetInstanceProcAddr) {
		VK_LOG("dlsym vkGetInstanceProcAddr failed");
		return -2;
	}
	// Only global entry points may be queried with VK_NULL_HANDLE.
	L->vkCreateInstance = (PFN_vkCreateInstance)L->vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
	if (!L->vkCreateInstance)
		L->vkCreateInstance = (PFN_vkCreateInstance)dlsym(L->lib, "vkCreateInstance");
	if (!L->vkCreateInstance)
		return -3;
	VK_LOG("Vulkan loader ready (CreateInstance)");
	return 0;
}

static void bk_vkLoadInstance(vkLoader *L, VkInstance inst) {
#define LOAD_I(name) L->name = (PFN_##name)L->vkGetInstanceProcAddr(inst, #name)
	LOAD_I(vkDestroyInstance);
	LOAD_I(vkEnumeratePhysicalDevices);
	LOAD_I(vkGetPhysicalDeviceQueueFamilyProperties);
	LOAD_I(vkGetPhysicalDeviceMemoryProperties);
	LOAD_I(vkGetPhysicalDeviceProperties);
	LOAD_I(vkCreateDevice);
	LOAD_I(vkGetDeviceProcAddr);
#undef LOAD_I
	if (!L->vkGetDeviceProcAddr)
		L->vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)dlsym(L->lib, "vkGetDeviceProcAddr");
}

static void bk_vkLoadDevice(vkLoader *L, VkInstance inst, VkDevice dev) {
#define LOAD_D(name) L->name = (PFN_##name)bk_vkSym(L, inst, dev, #name)
	LOAD_D(vkDestroyDevice);
	LOAD_D(vkGetDeviceQueue);
	LOAD_D(vkCreateCommandPool);
	LOAD_D(vkDestroyCommandPool);
	LOAD_D(vkAllocateCommandBuffers);
	LOAD_D(vkFreeCommandBuffers);
	LOAD_D(vkBeginCommandBuffer);
	LOAD_D(vkEndCommandBuffer);
	LOAD_D(vkCmdBindPipeline);
	LOAD_D(vkCmdBindDescriptorSets);
	LOAD_D(vkCmdDispatch);
	LOAD_D(vkCmdPipelineBarrier);
	LOAD_D(vkQueueSubmit);
	LOAD_D(vkQueueWaitIdle);
	LOAD_D(vkDeviceWaitIdle);
	LOAD_D(vkCreateBuffer);
	LOAD_D(vkDestroyBuffer);
	LOAD_D(vkGetBufferMemoryRequirements);
	LOAD_D(vkAllocateMemory);
	LOAD_D(vkFreeMemory);
	LOAD_D(vkBindBufferMemory);
	LOAD_D(vkMapMemory);
	LOAD_D(vkUnmapMemory);
	LOAD_D(vkCreateShaderModule);
	LOAD_D(vkDestroyShaderModule);
	LOAD_D(vkCreateDescriptorSetLayout);
	LOAD_D(vkDestroyDescriptorSetLayout);
	LOAD_D(vkCreatePipelineLayout);
	LOAD_D(vkDestroyPipelineLayout);
	LOAD_D(vkCreateComputePipelines);
	LOAD_D(vkDestroyPipeline);
	LOAD_D(vkCreateDescriptorPool);
	LOAD_D(vkDestroyDescriptorPool);
	LOAD_D(vkAllocateDescriptorSets);
	LOAD_D(vkUpdateDescriptorSets);
	LOAD_D(vkResetCommandPool);
	LOAD_D(vkCreateFence);
	LOAD_D(vkDestroyFence);
	LOAD_D(vkWaitForFences);
	LOAD_D(vkResetFences);
#undef LOAD_D
}

static vkLoader *bk_vkLoaderNew(void) {
	return (vkLoader *)calloc(1, sizeof(vkLoader));
}
static void bk_vkLoaderFree(vkLoader *L) {
	if (!L)
		return;
	if (L->lib)
		dlclose(L->lib);
	L->lib = NULL;
	free(L);
}

// Full device boot in C — avoids cgo "Go pointer to Go pointer" panics.
// Handle out-params are uintptr_t so Go never takes the address of a pointer-typed slot.
static int bk_vkBoot(vkLoader *L,
	uintptr_t *outInst, uintptr_t *outPD, uintptr_t *outDev, uintptr_t *outQ,
	uint32_t *outFam, char *nameOut, size_t nameCap,
	VkPhysicalDeviceMemoryProperties *outMem, uintptr_t *outPool) {
	if (bk_vkInitLoader(L) != 0)
		return -1;

	VkApplicationInfo app = {0};
	app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.pApplicationName = "birdkit";
	app.applicationVersion = 1;
	app.pEngineName = "welvet";
	app.engineVersion = 1;
	app.apiVersion = VK_API_VERSION_1_1;

	VkInstanceCreateInfo ici = {0};
	ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	ici.pApplicationInfo = &app;

	VkInstance inst = VK_NULL_HANDLE;
	VkResult r = L->vkCreateInstance(&ici, NULL, &inst);
	if (r != VK_SUCCESS) {
		VK_LOG("vkCreateInstance failed: %d", (int)r);
		return -2;
	}
	bk_vkLoadInstance(L, inst);
	if (!L->vkEnumeratePhysicalDevices || !L->vkCreateDevice) {
		VK_LOG("instance procs missing");
		if (L->vkDestroyInstance) L->vkDestroyInstance(inst, NULL);
		return -3;
	}

	uint32_t n = 0;
	L->vkEnumeratePhysicalDevices(inst, &n, NULL);
	if (n == 0) {
		L->vkDestroyInstance(inst, NULL);
		return -4;
	}
	VkPhysicalDevice physList[8];
	if (n > 8) n = 8;
	L->vkEnumeratePhysicalDevices(inst, &n, physList);
	VkPhysicalDevice pd = physList[0];

	VkPhysicalDeviceProperties props;
	L->vkGetPhysicalDeviceProperties(pd, &props);
	if (nameOut && nameCap > 0) {
		strncpy(nameOut, props.deviceName, nameCap - 1);
		nameOut[nameCap - 1] = 0;
	}

	uint32_t qCount = 0;
	L->vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, NULL);
	VkQueueFamilyProperties qfam[32];
	if (qCount > 32) qCount = 32;
	L->vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, qfam);
	uint32_t computeFam = 0;
	int found = 0;
	for (uint32_t i = 0; i < qCount; i++) {
		if (qfam[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			computeFam = i;
			found = 1;
			break;
		}
	}
	if (!found) {
		L->vkDestroyInstance(inst, NULL);
		return -5;
	}

	float priority = 1.0f;
	VkDeviceQueueCreateInfo qci = {0};
	qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	qci.queueFamilyIndex = computeFam;
	qci.queueCount = 1;
	qci.pQueuePriorities = &priority;

	VkDeviceCreateInfo dci = {0};
	dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;

	VkDevice dev = VK_NULL_HANDLE;
	r = L->vkCreateDevice(pd, &dci, NULL, &dev);
	if (r != VK_SUCCESS) {
		VK_LOG("vkCreateDevice failed: %d", (int)r);
		L->vkDestroyInstance(inst, NULL);
		return -6;
	}
	bk_vkLoadDevice(L, inst, dev);
	if (!L->vkCreateBuffer || !L->vkCmdDispatch) {
		VK_LOG("device procs missing");
		if (L->vkDestroyDevice) L->vkDestroyDevice(dev, NULL);
		L->vkDestroyInstance(inst, NULL);
		return -7;
	}

	VkQueue q = VK_NULL_HANDLE;
	L->vkGetDeviceQueue(dev, computeFam, 0, &q);

	VkCommandPoolCreateInfo cpci = {0};
	cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	cpci.queueFamilyIndex = computeFam;
	VkCommandPool pool = VK_NULL_HANDLE;
	r = L->vkCreateCommandPool(dev, &cpci, NULL, &pool);
	if (r != VK_SUCCESS) {
		L->vkDestroyDevice(dev, NULL);
		L->vkDestroyInstance(inst, NULL);
		return -8;
	}

	L->vkGetPhysicalDeviceMemoryProperties(pd, outMem);

	*outInst = (uintptr_t)inst;
	*outPD = (uintptr_t)pd;
	*outDev = (uintptr_t)dev;
	*outQ = (uintptr_t)q;
	*outFam = computeFam;
	*outPool = (uintptr_t)pool;
	VK_LOG("Vulkan device ready: %s", nameOut ? nameOut : "?");
	return 0;
}

static int bk_vkProbe(void) {
	void *lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
	if (!lib)
		return 0;
	dlclose(lib);
	return 1;
}

static uint32_t bk_vkFindMemoryType(const VkPhysicalDeviceMemoryProperties *mp, uint32_t typeBits, VkMemoryPropertyFlags props) {
	for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
		if ((typeBits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & props) == props)
			return i;
	}
	return 0;
}

static void bk_set_buf_info(VkDescriptorBufferInfo *i, VkBuffer b, VkDeviceSize off, VkDeviceSize r) {
	i->buffer = b;
	i->offset = off;
	i->range = r;
}

// All Vulkan handles cross the cgo boundary as uintptr_t. NDK Vulkan types many
// handles as pointers; opaque integer IDs cast to pointers can look like Go
// heap addresses and trip "Go pointer to unpinned Go pointer" during generate.

static void bk_vkComputeBarrier(vkLoader *L, uintptr_t cmdU) {
	VkCommandBuffer cmd = (VkCommandBuffer)cmdU;
	VkMemoryBarrier mb = {0};
	mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	L->vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 1, &mb, 0, NULL, 0, NULL);
}

static void bk_vkBindCompute(vkLoader *L, uintptr_t cmdU, uintptr_t pipeU, uintptr_t layoutU, uintptr_t setU) {
	VkCommandBuffer cmd = (VkCommandBuffer)cmdU;
	VkPipeline pipe = (VkPipeline)pipeU;
	VkPipelineLayout layout = (VkPipelineLayout)layoutU;
	VkDescriptorSet set = (VkDescriptorSet)setU;
	L->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
	L->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
}

static void bk_vkDispatch3(vkLoader *L, uintptr_t cmdU, uint32_t x, uint32_t y, uint32_t z) {
	L->vkCmdDispatch((VkCommandBuffer)cmdU, x, y, z);
}

// --- Go-callable wrappers (Go cannot invoke C function pointers directly) ---

static void bk_DestroyInstance(vkLoader *L, uintptr_t instU) {
	L->vkDestroyInstance((VkInstance)instU, NULL);
}
static void bk_DestroyDevice(vkLoader *L, uintptr_t devU) {
	L->vkDestroyDevice((VkDevice)devU, NULL);
}
static void bk_DestroyCommandPool(vkLoader *L, uintptr_t devU, uintptr_t poolU) {
	L->vkDestroyCommandPool((VkDevice)devU, (VkCommandPool)poolU, NULL);
}
static VkResult bk_CreateBufferHost(vkLoader *L, uintptr_t devU, const VkPhysicalDeviceMemoryProperties *mp,
	VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps,
	uintptr_t *outBuf, uintptr_t *outMem, uintptr_t *mappedOut) {
	VkDevice dev = (VkDevice)devU;
	VkBufferCreateInfo bci = {0};
	bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bci.size = size;
	bci.usage = usage;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkBuffer buf = VK_NULL_HANDLE;
	VkResult r = L->vkCreateBuffer(dev, &bci, NULL, &buf);
	if (r != VK_SUCCESS)
		return r;
	VkMemoryRequirements req;
	L->vkGetBufferMemoryRequirements(dev, buf, &req);
	VkMemoryAllocateInfo mai = {0};
	mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	mai.allocationSize = req.size;
	mai.memoryTypeIndex = bk_vkFindMemoryType(mp, req.memoryTypeBits, memProps);
	VkDeviceMemory mem = VK_NULL_HANDLE;
	r = L->vkAllocateMemory(dev, &mai, NULL, &mem);
	if (r != VK_SUCCESS)
		return r;
	r = L->vkBindBufferMemory(dev, buf, mem, 0);
	if (r != VK_SUCCESS)
		return r;
	if (mappedOut)
		*mappedOut = 0;
	if (mappedOut && (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
		void *p = NULL;
		r = L->vkMapMemory(dev, mem, 0, mai.allocationSize, 0, &p);
		if (r == VK_SUCCESS)
			*mappedOut = (uintptr_t)p;
	}
	*outBuf = (uintptr_t)buf;
	*outMem = (uintptr_t)mem;
	return r;
}
static void bk_DestroyBuffer(vkLoader *L, uintptr_t devU, uintptr_t bufU) {
	L->vkDestroyBuffer((VkDevice)devU, (VkBuffer)bufU, NULL);
}
static void bk_UnmapMemory(vkLoader *L, uintptr_t devU, uintptr_t memU) {
	L->vkUnmapMemory((VkDevice)devU, (VkDeviceMemory)memU);
}
static void bk_FreeMemory(vkLoader *L, uintptr_t devU, uintptr_t memU) {
	L->vkFreeMemory((VkDevice)devU, (VkDeviceMemory)memU, NULL);
}
static VkResult bk_CreateComputePipe(vkLoader *L, uintptr_t devU,
	const uint32_t *code, size_t codeBytes, uint32_t numBindings,
	const uint32_t *descTypes,
	uintptr_t *outSL, uintptr_t *outPL, uintptr_t *outPipe) {
	VkDevice dev = (VkDevice)devU;
	if (numBindings == 0 || numBindings > 16 || !descTypes)
		return VK_ERROR_INITIALIZATION_FAILED;
	VkDescriptorSetLayoutBinding bindings[16];
	memset(bindings, 0, sizeof(bindings));
	for (uint32_t i = 0; i < numBindings; i++) {
		bindings[i].binding = i;
		bindings[i].descriptorType = (VkDescriptorType)descTypes[i];
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo dslci = {0};
	dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dslci.bindingCount = numBindings;
	dslci.pBindings = bindings;
	VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
	VkResult r = L->vkCreateDescriptorSetLayout(dev, &dslci, NULL, &setLayout);
	if (r != VK_SUCCESS)
		return r;
	VkPipelineLayoutCreateInfo plci = {0};
	plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plci.setLayoutCount = 1;
	plci.pSetLayouts = &setLayout;
	VkPipelineLayout layout = VK_NULL_HANDLE;
	r = L->vkCreatePipelineLayout(dev, &plci, NULL, &layout);
	if (r != VK_SUCCESS) {
		L->vkDestroyDescriptorSetLayout(dev, setLayout, NULL);
		return r;
	}
	VkShaderModuleCreateInfo smi = {0};
	smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	smi.codeSize = codeBytes;
	smi.pCode = code;
	VkShaderModule mod = VK_NULL_HANDLE;
	r = L->vkCreateShaderModule(dev, &smi, NULL, &mod);
	if (r != VK_SUCCESS) {
		L->vkDestroyPipelineLayout(dev, layout, NULL);
		L->vkDestroyDescriptorSetLayout(dev, setLayout, NULL);
		return r;
	}
	VkPipelineShaderStageCreateInfo stage = {0};
	stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = mod;
	stage.pName = "main";
	VkComputePipelineCreateInfo cpci = {0};
	cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	cpci.stage = stage;
	cpci.layout = layout;
	VkPipeline pipe = VK_NULL_HANDLE;
	r = L->vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe);
	L->vkDestroyShaderModule(dev, mod, NULL);
	if (r != VK_SUCCESS) {
		VK_LOG("CreateComputePipelines failed: %d", (int)r);
		L->vkDestroyPipelineLayout(dev, layout, NULL);
		L->vkDestroyDescriptorSetLayout(dev, setLayout, NULL);
		return r;
	}
	*outSL = (uintptr_t)setLayout;
	*outPL = (uintptr_t)layout;
	*outPipe = (uintptr_t)pipe;
	return r;
}
static void bk_DestroyDescriptorSetLayout(vkLoader *L, uintptr_t devU, uintptr_t slU) {
	L->vkDestroyDescriptorSetLayout((VkDevice)devU, (VkDescriptorSetLayout)slU, NULL);
}
static void bk_DestroyPipelineLayout(vkLoader *L, uintptr_t devU, uintptr_t plU) {
	L->vkDestroyPipelineLayout((VkDevice)devU, (VkPipelineLayout)plU, NULL);
}
static void bk_DestroyPipeline(vkLoader *L, uintptr_t devU, uintptr_t pipeU) {
	L->vkDestroyPipeline((VkDevice)devU, (VkPipeline)pipeU, NULL);
}
static void bk_DestroyDescriptorPool(vkLoader *L, uintptr_t devU, uintptr_t poolU) {
	L->vkDestroyDescriptorPool((VkDevice)devU, (VkDescriptorPool)poolU, NULL);
}
static void bk_WriteDescBuffers(vkLoader *L, uintptr_t devU, uintptr_t setU, uint32_t n,
	const uintptr_t *bufs, const VkDeviceSize *offs, const VkDeviceSize *ranges,
	const uint32_t *descTypes) {
	if (n == 0 || n > 16 || !descTypes || !bufs)
		return;
	VkDevice dev = (VkDevice)devU;
	VkDescriptorSet set = (VkDescriptorSet)setU;
	VkDescriptorBufferInfo infos[16];
	VkWriteDescriptorSet writes[16];
	memset(infos, 0, sizeof(infos));
	memset(writes, 0, sizeof(writes));
	for (uint32_t i = 0; i < n; i++) {
		infos[i].buffer = (VkBuffer)bufs[i];
		infos[i].offset = offs[i];
		infos[i].range = ranges[i];
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = set;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = (VkDescriptorType)descTypes[i];
		writes[i].pBufferInfo = &infos[i];
	}
	L->vkUpdateDescriptorSets(dev, n, writes, 0, NULL);
}
static VkResult bk_AllocOneDescSet(vkLoader *L, uintptr_t devU, uintptr_t poolU,
	uintptr_t layoutU, uintptr_t *out) {
	VkDevice dev = (VkDevice)devU;
	VkDescriptorSetLayout layout = (VkDescriptorSetLayout)layoutU;
	VkDescriptorSetAllocateInfo ai = {0};
	ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool = (VkDescriptorPool)poolU;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts = &layout;
	VkDescriptorSet set = VK_NULL_HANDLE;
	VkResult r = L->vkAllocateDescriptorSets(dev, &ai, &set);
	if (r == VK_SUCCESS && out)
		*out = (uintptr_t)set;
	return r;
}
static void bk_ResetCommandPool(vkLoader *L, uintptr_t devU, uintptr_t poolU) {
	L->vkResetCommandPool((VkDevice)devU, (VkCommandPool)poolU, 0);
}
static VkResult bk_AllocOneCmd(vkLoader *L, uintptr_t devU, uintptr_t poolU, uintptr_t *out) {
	VkCommandBufferAllocateInfo ai = {0};
	ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.commandPool = (VkCommandPool)poolU;
	ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = 1;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	VkResult r = L->vkAllocateCommandBuffers((VkDevice)devU, &ai, &cmd);
	if (r == VK_SUCCESS && out)
		*out = (uintptr_t)cmd;
	return r;
}
static void bk_FreeCommandBuffers(vkLoader *L, uintptr_t devU, uintptr_t poolU, uintptr_t cmdU) {
	VkCommandBuffer cmd = (VkCommandBuffer)cmdU;
	L->vkFreeCommandBuffers((VkDevice)devU, (VkCommandPool)poolU, 1, &cmd);
}
static VkResult bk_BeginOneTime(vkLoader *L, uintptr_t cmdU) {
	VkCommandBuffer cmd = (VkCommandBuffer)cmdU;
	VkCommandBufferBeginInfo bi = {0};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	return L->vkBeginCommandBuffer(cmd, &bi);
}
static VkResult bk_EndCommandBuffer(vkLoader *L, uintptr_t cmdU) {
	return L->vkEndCommandBuffer((VkCommandBuffer)cmdU);
}
static VkResult bk_CreateFence(vkLoader *L, uintptr_t devU, uintptr_t *out) {
	VkFenceCreateInfo ci = {0};
	ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	VkFence fence = VK_NULL_HANDLE;
	VkResult r = L->vkCreateFence((VkDevice)devU, &ci, NULL, &fence);
	if (r == VK_SUCCESS && out)
		*out = (uintptr_t)fence;
	return r;
}
static void bk_DestroyFence(vkLoader *L, uintptr_t devU, uintptr_t fenceU) {
	L->vkDestroyFence((VkDevice)devU, (VkFence)fenceU, NULL);
}
static VkResult bk_QueueSubmit(vkLoader *L, uintptr_t qU, uintptr_t cmdU, uintptr_t fenceU) {
	VkCommandBuffer cmd = (VkCommandBuffer)cmdU;
	VkSubmitInfo si = {0};
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd;
	return L->vkQueueSubmit((VkQueue)qU, 1, &si, (VkFence)fenceU);
}
static VkResult bk_WaitForFences(vkLoader *L, uintptr_t devU, uintptr_t fenceU) {
	VkFence fence = (VkFence)fenceU;
	return L->vkWaitForFences((VkDevice)devU, 1, &fence, VK_TRUE, (uint64_t)1e10);
}

static VkResult bk_CreateDescPool(vkLoader *L, uintptr_t devU, uint32_t maxSets, uint32_t ub, uint32_t sb, uintptr_t *out) {
	VkDescriptorPoolSize sizes[2];
	sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	sizes[0].descriptorCount = ub;
	sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	sizes[1].descriptorCount = sb;
	VkDescriptorPoolCreateInfo ci = {0};
	ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	ci.maxSets = maxSets;
	ci.poolSizeCount = 2;
	ci.pPoolSizes = sizes;
	VkDescriptorPool pool = VK_NULL_HANDLE;
	VkResult r = L->vkCreateDescriptorPool((VkDevice)devU, &ci, NULL, &pool);
	if (r == VK_SUCCESS && out)
		*out = (uintptr_t)pool;
	return r;
}
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// ErrNotAvailable is returned when native Vulkan hybrid is unavailable.
var ErrNotAvailable = errors.New("fusedgpu: native Vulkan hybrid unavailable")

func vkAlignSize(n uint64) uint64 {
	if n < 64 {
		n = 64
	}
	if n%16 != 0 {
		n = (n + 15) &^ 15
	}
	return n
}

type vkDevice struct {
	ld          *C.vkLoader // C heap — never pass &GoLoader to cgo
	inst        uintptr
	phys        uintptr
	dev         uintptr
	queue       uintptr
	cmdPool     uintptr
	descPool    uintptr
	queueFamily uint32
	name        string
	maxSSBO     uint64
	memProps    C.VkPhysicalDeviceMemoryProperties
}

type vkBuffer struct {
	dev     *vkDevice
	buf     uintptr
	mem     uintptr
	size    uint64
	mapped  unsafe.Pointer
	staging bool
}

type vkPipelineBundle struct {
	layout    uintptr
	setLayout uintptr
	pipe      uintptr
	bindings  int
	types     []C.uint32_t // VkDescriptorType per binding
}

func NativeVKAvailable() bool {
	return C.bk_vkProbe() != 0
}

func vkALog(msg string) {
	c := C.CString(msg)
	defer C.free(unsafe.Pointer(c))
	C.bk_ALog(c)
}

func initVKDevice() (*vkDevice, error) {
	ld := C.bk_vkLoaderNew()
	if ld == nil {
		return nil, fmt.Errorf("vulkan loader alloc failed")
	}
	var instU, pdU, devU, qU, poolU C.uintptr_t
	var fam C.uint32_t
	var memProps C.VkPhysicalDeviceMemoryProperties
	nameBuf := make([]byte, 256)
	rc := C.bk_vkBoot(ld, &instU, &pdU, &devU, &qU, &fam,
		(*C.char)(unsafe.Pointer(&nameBuf[0])), C.size_t(len(nameBuf)),
		&memProps, &poolU)
	if rc != 0 {
		C.bk_vkLoaderFree(ld)
		return nil, fmt.Errorf("vulkan boot: %d", int(rc))
	}
	name := C.GoString((*C.char)(unsafe.Pointer(&nameBuf[0])))
	return &vkDevice{
		ld:          ld,
		inst:        uintptr(instU),
		phys:        uintptr(pdU),
		dev:         uintptr(devU),
		queue:       uintptr(qU),
		cmdPool:     uintptr(poolU),
		queueFamily: uint32(fam),
		name:        name,
		maxSSBO:     1 << 30,
		memProps:    memProps,
	}, nil
}

func (d *vkDevice) createBuffer(label string, size uint64, storage, hostVisible bool, data []byte) (*vkBuffer, error) {
	_ = label
	size = vkAlignSize(size)
	usage := C.VkBufferUsageFlags(C.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | C.VK_BUFFER_USAGE_TRANSFER_DST_BIT)
	if storage {
		usage |= C.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
	} else {
		usage |= C.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
	}
	memProps := C.VkMemoryPropertyFlags(C.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | C.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
	if !hostVisible {
		memProps = C.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
	}
	var bufU, memU, mapped C.uintptr_t
	if r := C.bk_CreateBufferHost(d.ld, C.uintptr_t(d.dev), &d.memProps, C.VkDeviceSize(size), usage, memProps,
		&bufU, &memU, &mapped); r != C.VK_SUCCESS {
		return nil, fmt.Errorf("createBuffer: %d", int(r))
	}
	mappedPtr := unsafe.Pointer(uintptr(mapped))
	vb := &vkBuffer{
		dev: d, buf: uintptr(bufU), mem: uintptr(memU),
		size: size, mapped: mappedPtr, staging: hostVisible,
	}
	if len(data) > 0 && hostVisible && mappedPtr != nil {
		copy(unsafe.Slice((*byte)(mappedPtr), len(data)), data)
	}
	return vb, nil
}

func (b *vkBuffer) destroy() {
	if b == nil || b.dev == nil || b.dev.ld == nil {
		return
	}
	ld := b.dev.ld
	if b.mapped != nil {
		C.bk_UnmapMemory(ld, C.uintptr_t(b.dev.dev), C.uintptr_t(b.mem))
		b.mapped = nil
	}
	C.bk_DestroyBuffer(ld, C.uintptr_t(b.dev.dev), C.uintptr_t(b.buf))
	C.bk_FreeMemory(ld, C.uintptr_t(b.dev.dev), C.uintptr_t(b.mem))
}

func (d *vkDevice) createComputePipeline(name string, spirv []byte, descTypes []C.uint32_t) (*vkPipelineBundle, error) {
	if len(spirv) < 4 || len(spirv)%4 != 0 {
		return nil, fmt.Errorf("pipeline %s: bad spirv", name)
	}
	if len(descTypes) == 0 || len(descTypes) > 16 {
		return nil, fmt.Errorf("pipeline %s: bad descriptor layout", name)
	}
	cCode := C.CBytes(spirv)
	defer C.free(cCode)
	typesCopy := make([]C.uint32_t, len(descTypes))
	copy(typesCopy, descTypes)
	var slU, plU, pipeU C.uintptr_t
	if r := C.bk_CreateComputePipe(d.ld, C.uintptr_t(d.dev),
		(*C.uint32_t)(cCode), C.size_t(len(spirv)), C.uint32_t(len(typesCopy)),
		&typesCopy[0],
		&slU, &plU, &pipeU); r != C.VK_SUCCESS {
		return nil, fmt.Errorf("%s pipeline: %d", name, int(r))
	}
	return &vkPipelineBundle{
		layout: uintptr(plU), setLayout: uintptr(slU),
		pipe:     uintptr(pipeU),
		bindings: len(typesCopy), types: typesCopy,
	}, nil
}

func (d *vkDevice) allocDescPool(maxSets, ub, sb int) error {
	var poolU C.uintptr_t
	if r := C.bk_CreateDescPool(d.ld, C.uintptr_t(d.dev), C.uint32_t(maxSets), C.uint32_t(ub), C.uint32_t(sb), &poolU); r != C.VK_SUCCESS {
		return fmt.Errorf("descriptor pool: %d", int(r))
	}
	d.descPool = uintptr(poolU)
	return nil
}

func (d *vkDevice) writeDescriptorSet(set uintptr, bufs []*vkBuffer, offsets []uint64, descTypes []C.uint32_t) {
	n := len(bufs)
	if n == 0 {
		return
	}
	if len(descTypes) < n {
		return
	}
	cBufs := make([]C.uintptr_t, n)
	cOffs := make([]C.VkDeviceSize, n)
	cRanges := make([]C.VkDeviceSize, n)
	for i, b := range bufs {
		off := uint64(0)
		if i < len(offsets) {
			off = offsets[i]
		}
		rng := b.size - off
		if rng == 0 {
			rng = b.size
		}
		cBufs[i] = C.uintptr_t(b.buf)
		cOffs[i] = C.VkDeviceSize(off)
		cRanges[i] = C.VkDeviceSize(rng)
	}
	// Copy descriptor types onto C heap so cgo never sees a Go slice pointer
	// that the checker might walk conservatively.
	cTypes := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.uint32_t(0))))
	if cTypes == nil {
		return
	}
	defer C.free(cTypes)
	typesSlice := unsafe.Slice((*C.uint32_t)(cTypes), n)
	copy(typesSlice, descTypes[:n])
	C.bk_WriteDescBuffers(d.ld, C.uintptr_t(d.dev), C.uintptr_t(set), C.uint32_t(n),
		&cBufs[0], &cOffs[0], &cRanges[0], (*C.uint32_t)(cTypes))
}

func (d *vkDevice) allocDescSet(layout uintptr) (uintptr, error) {
	var setU C.uintptr_t
	if r := C.bk_AllocOneDescSet(d.ld, C.uintptr_t(d.dev), C.uintptr_t(d.descPool), C.uintptr_t(layout), &setU); r != C.VK_SUCCESS {
		return 0, fmt.Errorf("allocDescSet: %d", int(r))
	}
	return uintptr(setU), nil
}

func (d *vkDevice) computeBarrier(cmd uintptr) {
	C.bk_vkComputeBarrier(d.ld, C.uintptr_t(cmd))
}

func (d *vkDevice) bindDispatch(cmd, pipe, layout, set uintptr, x, y, z uint32) {
	C.bk_vkBindCompute(d.ld, C.uintptr_t(cmd), C.uintptr_t(pipe), C.uintptr_t(layout), C.uintptr_t(set))
	C.bk_vkDispatch3(d.ld, C.uintptr_t(cmd), C.uint32_t(x), C.uint32_t(y), C.uint32_t(z))
}

func (d *vkDevice) submitDispatches(rec func(cmd uintptr)) error {
	ld := d.ld
	C.bk_ResetCommandPool(ld, C.uintptr_t(d.dev), C.uintptr_t(d.cmdPool))
	var cmdU C.uintptr_t
	if r := C.bk_AllocOneCmd(ld, C.uintptr_t(d.dev), C.uintptr_t(d.cmdPool), &cmdU); r != C.VK_SUCCESS {
		return fmt.Errorf("alloc cmd: %d", int(r))
	}
	cmd := uintptr(cmdU)
	defer C.bk_FreeCommandBuffers(ld, C.uintptr_t(d.dev), C.uintptr_t(d.cmdPool), C.uintptr_t(cmd))

	if r := C.bk_BeginOneTime(ld, C.uintptr_t(cmd)); r != C.VK_SUCCESS {
		return fmt.Errorf("begin cmd: %d", int(r))
	}
	rec(cmd)
	if r := C.bk_EndCommandBuffer(ld, C.uintptr_t(cmd)); r != C.VK_SUCCESS {
		return fmt.Errorf("end cmd: %d", int(r))
	}

	var fenceU C.uintptr_t
	if r := C.bk_CreateFence(ld, C.uintptr_t(d.dev), &fenceU); r != C.VK_SUCCESS {
		return fmt.Errorf("fence: %d", int(r))
	}
	defer C.bk_DestroyFence(ld, C.uintptr_t(d.dev), fenceU)

	if r := C.bk_QueueSubmit(ld, C.uintptr_t(d.queue), C.uintptr_t(cmd), fenceU); r != C.VK_SUCCESS {
		return fmt.Errorf("submit: %s", vkResultString(r))
	}
	if r := C.bk_WaitForFences(ld, C.uintptr_t(d.dev), fenceU); r != C.VK_SUCCESS {
		return fmt.Errorf("wait: %s", vkResultString(r))
	}
	return nil
}

func vkResultString(r C.VkResult) string {
	switch r {
	case C.VK_SUCCESS:
		return "VK_SUCCESS"
	case C.VK_ERROR_DEVICE_LOST:
		return "VK_ERROR_DEVICE_LOST (-4)"
	case C.VK_ERROR_OUT_OF_DEVICE_MEMORY:
		return "VK_ERROR_OUT_OF_DEVICE_MEMORY"
	case C.VK_ERROR_OUT_OF_HOST_MEMORY:
		return "VK_ERROR_OUT_OF_HOST_MEMORY"
	case C.VK_TIMEOUT:
		return "VK_TIMEOUT"
	default:
		return fmt.Sprintf("VkResult(%d)", int(r))
	}
}

func (d *vkDevice) destroyPipelineBundle(p *vkPipelineBundle) {
	if p == nil || d == nil || d.ld == nil {
		return
	}
	ld := d.ld
	if p.pipe != 0 {
		C.bk_DestroyPipeline(ld, C.uintptr_t(d.dev), C.uintptr_t(p.pipe))
	}
	if p.layout != 0 {
		C.bk_DestroyPipelineLayout(ld, C.uintptr_t(d.dev), C.uintptr_t(p.layout))
	}
	if p.setLayout != 0 {
		C.bk_DestroyDescriptorSetLayout(ld, C.uintptr_t(d.dev), C.uintptr_t(p.setLayout))
	}
}

func (d *vkDevice) destroy() {
	if d == nil {
		return
	}
	ld := d.ld
	if ld != nil {
		if d.descPool != 0 {
			C.bk_DestroyDescriptorPool(ld, C.uintptr_t(d.dev), C.uintptr_t(d.descPool))
			d.descPool = 0
		}
		if d.cmdPool != 0 {
			C.bk_DestroyCommandPool(ld, C.uintptr_t(d.dev), C.uintptr_t(d.cmdPool))
			d.cmdPool = 0
		}
		if d.dev != 0 {
			C.bk_DestroyDevice(ld, C.uintptr_t(d.dev))
			d.dev = 0
		}
		if d.inst != 0 {
			C.bk_DestroyInstance(ld, C.uintptr_t(d.inst))
			d.inst = 0
		}
		C.bk_vkLoaderFree(ld)
		d.ld = nil
	}
}

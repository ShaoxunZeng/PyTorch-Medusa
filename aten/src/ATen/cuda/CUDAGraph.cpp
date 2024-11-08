#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/cuda/CUDAContext.h>

#include <chrono>
#include <thread>
#include <set>
#include <fstream>
#include <iostream>
#include <regex>

#include <cuda.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <link.h>

namespace at::cuda {

static bool _cuda_graphs_debug = false;
static bool _cuda_graphs_save = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

MempoolId_t graph_pool_handle() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by graph_pool_handle".
  static std::atomic<CaptureId_t> uid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return {0, uid++};
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
  return {0, 0};
#endif
}


// Get the expected id of a capture sequence so that we can call beginAllocateStreamToPool
// before starting a graph capture
CaptureId_t capture_sequence_id() {
  // id starts at 1:
  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by cudaStreamGetCaptureInfo".
  // (But how do we know GetCaptureInfo never sets id_ to 0? Because that's the current behavior,
  // and I asked cuda devs to keep it that way, and they agreed.)
  static std::atomic<CaptureId_t> uuid{1};
  return uuid++;
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

std::atomic<int> CUDAGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL watchdog so that they
// can be resolved before the capture begins. Note that event queries are not allowed during a
// graph capture in the default capture mode.
void CUDAGraph::inc_pending_event_queries() {
  pending_event_queries++;
}

void CUDAGraph::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(pending_event_queries > 0,
    "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
  pending_event_queries--;
}

int CUDAGraph::num_pending_event_queries() {
  return pending_event_queries;
}

void bind_core(int core_no) {
  cpu_set_t cpu_mask;
  CPU_ZERO(&cpu_mask);
  CPU_SET(core_no, &cpu_mask);

  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask) != 0) {
    std::cerr << "Failed to set thread affinity mask." << std::endl;
    exit(-1);
  }
}

CUDAGraph::CUDAGraph()
  // CUDAStreams may not be default-constructed.
  : capture_stream_(at::cuda::getCurrentCUDAStream()) {
#if (defined(USE_ROCM) && ROCM_VERSION < 50300)
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3");
#endif
}

void CUDAGraph::set_info_files(const std::string &dependency_file, const std::string &func_params_file, const std::string &node_file, const std::string &mf_file) {
  SERVERLESS_LOG("dependency_file = %s\n", dependency_file.c_str());
  SERVERLESS_LOG("func_params_file = %s\n", func_params_file.c_str());
  SERVERLESS_LOG("node_file = %s\n", node_file.c_str());
  SERVERLESS_LOG("mf_file = %s\n", mf_file.c_str());

  if (dependency_file.empty() || func_params_file.empty() || node_file.empty() || mf_file.empty()) {
    TORCH_CHECK(false, "missing some files, can not read\n");
    return;
  }

  dependency_file_stream = std::ifstream(dependency_file);
  func_params_file_stream = std::ifstream(func_params_file);
  node_file_stream = std::ifstream(node_file);
  mf_file_stream = std::ifstream(mf_file);
}

void CUDAGraph::capture_begin(MempoolId_t pool/*=0*/, cudaStreamCaptureMode capture_mode, bool load_graph) {
  load_graph_ = load_graph;
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");
  SERVERLESS_LOG("=============== capture_begin ===============\n");

  // For now, a CUDAGraph instance only accommodates the default generator on the device that's
  // current when capture begins. If any op in the captured region uses a non-default generator,
  // or a generator on another device, the offending generator will throw an error.
  // These restrictions simplify CUDAGraph, but could be relaxed in the future:
  // in principle, the underlying Cuda calls do permit cross-device ops to be captured.
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());

  auto options = TensorOptions().device(at::kCUDA).dtype(at::kLong);
  seed_extragraph_ = at::empty({1}, options);
  offset_extragraph_ = at::empty({1}, options);

  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  gen->capture_prologue(seed_extragraph_.data_ptr<int64_t>(), offset_extragraph_.mutable_data_ptr<int64_t>());

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_gen_ = gen;
  capture_dev_ = c10::cuda::current_device();

  id_ = capture_sequence_id();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our mempool_id_.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10::cuda::CUDACachingAllocator::beginAllocateStreamToPool(capture_dev_, capture_stream_, mempool_id_);

  // At this point, any NCCL watchdogs should be aware that we are in capture mode
  // and therefore should not enqueue any additional work that could be event-queried.
  // We still must wait on any existing work that has not been cleaned up.
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE("Waiting for pending NCCL work to finish before starting graph capture.");
    std::this_thread::sleep_for(
      std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));

  cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  TORCH_INTERNAL_ASSERT(id_ > 0);
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

void CUDAGraph::capture_end() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));

  if(_cuda_graphs_save) {
    save_cuda_graph();
  }

  c10::cuda::CUDACachingAllocator::endAllocateStreamToPool(capture_dev_, capture_stream_);

  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
  has_graph_ = true;

  // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
  // between replays.
  // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
  // is cudaMallocAsync.
  // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
  // the graph's internal bookkeeping requires that we instantiate with
  // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
  // cudaGraphLaunch
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597
  // cudaGraphInstantiateWithFlags
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ga2c652a24ba93e52b99a47bec0888233
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
    // Trailing NULL, NULL, 0 arguments were recommended by Cuda driver people,
    // who prefer not to report error message through these arguments moving forward
    // (they prefer return value, or errors on api calls internal to the capture)
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
#else
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  } else {
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
  }
#endif

  has_graph_exec_ = true;

  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  TORCH_CHECK(gen == capture_gen_,
              "Default CUDA RNG generator on current device at capture end "
              "is different from default generator on current device "
              "when capture began");
  wholegraph_increment_ = gen->capture_epilogue();

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, NULL, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
      TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  // check if debug path is set
  if (!_cuda_graphs_debug) {
    // Now that we've instantiated graph_ into graph_exec_,
    // we don't need graph_ anymore.
    AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    has_graph_ = false;
  } else {
    TORCH_WARN("DEBUG: TORCH_CUDAGRAPHS_DEBUG_PATH detected. graph_ will not be freed until debug_dump is called.");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
  SERVERLESS_LOG("=============== capture_end ===============\n");
}

void CUDAGraph::replay() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  // Just like any RNG consumer kernel!
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(wholegraph_increment_);
  }
  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));

  // graph_exec_ may be replayed in any stream.
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    // Workaround for bug in libcuda.so that causes replayed graphs with
    // certain topologies to be corrupted (kernels elided, internal syncs
    // ignored) when replayed back to back without a sync in between.
    // The bug is fixed in CUDA 11.4+.
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#else
  TORCH_CHECK(false, "CUDA graphs is not yet supported on ROCM");
#endif
}

void CUDAGraph::enable_debug_mode() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  _cuda_graphs_debug = true;
#else
  TORCH_CHECK(false, "CUDA graphs is not yet supported on ROCM");
#endif

}

void CUDAGraph::debug_dump(const std::string& debug_path) {
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11030)|| (defined(USE_ROCM) && ROCM_VERSION >= 50600)
  if (_cuda_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      C10_CUDA_CHECK_WARN(cudaGraphDebugDotPrint(graph_, debug_path.c_str(), 1<<10)); // most verbose output
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    }
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with torch._C._cuda_enable_graphs_debug_mode");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

void CUDAGraph::enable_save_mode() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  _cuda_graphs_save = true;
#else
  TORCH_CHECK(false, "CUDA graphs is not yet supported on ROCM");
#endif
}

void CUDAGraph::set_model(const std::string& model_name) {
  if (model_name == "Llama-7B") {
    model = Llama2_7B;
    key_offset = 8192;
    value_offset = 16384;
  } else if (model_name == "Llama-13B") {
    model = Llama2_13B;
    key_offset = 10240;
    value_offset = 20480;
  } else if (model_name == "Yi-6B") {
    model = Yi_6B;
    key_offset = 8192;
    value_offset = 8192 + 1024;
  } else if (model_name == "Yi-9B") {
    model = Yi_9B;
    key_offset = 8192;
    value_offset = 8192 + 1024;
  } else if (model_name == "Qwen-14B") {
    model = Qwen_14B;
    key_offset = 10240;
    value_offset = key_offset + 10240;
  } else if (model_name == "Qwen-7B") {
    model = Qwen_7B;
    key_offset = 8192;
    value_offset = key_offset + 8192;
  } else if (model_name == "Qwen-4B") {
    model = Qwen_4B;
    key_offset = 5120;
    value_offset = key_offset + 5120;
  } else if (model_name == "Qwen-1.8B") {
    model = Qwen_1_8B;
    key_offset = 4096;
    value_offset = key_offset + 4096;
  } else if (model_name == "Qwen-0.5B") {
    model = Qwen_0_5B;
    key_offset = 2048;
    value_offset = key_offset + 2048;
  } else if (model_name == "Falcon-7B") {
    model = Falcon_7B;
    key_offset = 9088;
    value_offset = key_offset + 128;
  } else {
    TORCH_CHECK(false, "model name not supported");
  }
}

#define MAX_NODE_AND_EDGE_NUM 30000

bool found_lib;

int dl_iterate_phdr_callback(struct dl_phdr_info* info, size_t size, void* data) {
  void* symbol = dlsym(RTLD_DEFAULT, (char*)data);
  if (symbol != nullptr) {
    SERVERLESS_LOG("found library name: %s\n", info->dlpi_name);
    found_lib = true;
  }

  return 0;
}

void CUDAGraph::printTensorAddr(const at::Tensor &tensor) {
  SERVERLESS_LOG("tensor addr = %p\n", tensor.data_ptr());
}

void isCpuPointer(void* ptr) {
  cudaPointerAttributes gpuAttributes;

  cudaError_t gpuResult = cudaPointerGetAttributes(&gpuAttributes, ptr);

  if (gpuResult == cudaSuccess) {
    SERVERLESS_LOG("GPU Pointer\n");
  } else {
    SERVERLESS_LOG("Not a GPU Pointer\n");
  }
}

void printHexData(unsigned char* data, size_t length) {
  for (std::size_t i = 0; i < length; ++i) {
      SERVERLESS_LOG("%02x ", data[i]);
      if ((i + 1) % 16 == 0 && (i + 1) < length) {
          SERVERLESS_LOG("\n");
      }
  }
  SERVERLESS_LOG("\n");
}

void saveGraphKernelNodeParam(const CUfunction &func, void* kernelParams) {
  size_t paramIndex;
  size_t paramOffset;
  size_t paramSize;

  int ret;
  for (paramIndex = 0; ; paramIndex++) {
    ret = cuFuncGetParamInfo(func, paramIndex, &paramOffset, &paramSize);
    if (ret != CUDA_SUCCESS) {
      break;
    }
    SERVERLESS_LOG("paramIndex = %ld, paramOffset = %ld, paramSize = %ld\n", paramIndex, paramOffset, paramSize);
    printHexData((unsigned char*)(*(void**)((char*)kernelParams + paramIndex * 8)), paramSize);
    SERVERLESS_LOG("\n");
  }
}

void CUDAGraph::saveGraphKernelNodeParams(const CUDA_KERNEL_NODE_PARAMS &pNodeParams) {
  const char *name;
  CU_CALL(cuFuncGetName(&name, pNodeParams.func));
  SERVERLESS_LOG("kernel_func.name mangled = %s\n", name);

  SERVERLESS_LOG("kernel_params.blockDimX = %d\n", pNodeParams.blockDimX);
  SERVERLESS_LOG("kernel_params.blockDimY = %d\n", pNodeParams.blockDimY);
  SERVERLESS_LOG("kernel_params.blockDimZ = %d\n", pNodeParams.blockDimZ);
  SERVERLESS_LOG("kernel_params.ctx = %p\n", pNodeParams.ctx);
  SERVERLESS_LOG("kernel_params.extra = %p\n", pNodeParams.extra);
  SERVERLESS_LOG("kernel_params.func = %p\n", pNodeParams.func);
  SERVERLESS_LOG("kernel_params.gridDimX = %d\n", pNodeParams.gridDimX);
  SERVERLESS_LOG("kernel_params.gridDimY = %d\n", pNodeParams.gridDimY);
  SERVERLESS_LOG("kernel_params.gridDimZ = %d\n", pNodeParams.gridDimZ);
  SERVERLESS_LOG("kernel_params.kernel = %p\n", pNodeParams.kern);
  SERVERLESS_LOG("kernel_params.kernelParams = %p\n", pNodeParams.kernelParams);
  SERVERLESS_LOG("kernel_params.sharedMemBytes = %d\n", pNodeParams.sharedMemBytes);
  saveGraphKernelNodeParam(pNodeParams.func, pNodeParams.kernelParams);
}

void CUDAGraph::reuseGraphKernelNodeParams(int node_idx, const CUDA_KERNEL_NODE_PARAMS &p) {
  const char *name;
  CU_CALL(cuFuncGetName(&name, p.func));

  func2ptr_.insert(std::make_pair(name, p.func));

  SERVERLESS_LOG("save_func_ptrs %s\n", name);

  if (strcmp(name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x128x32_stage4_warpsize2x2x1_tensor16x8x16_kernel") == 0
    || strcmp(name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize192x128x32_stage3_warpsize4x2x1_tensor16x8x16_kernel") == 0
    || strcmp(name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x96x64_stage4_warpsize2x2x1_tensor16x8x16_kernel") == 0
    || strcmp(name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x64x64_stage4_warpsize2x2x1_tensor16x8x16_kernel") == 0) {
  
    int* addr1 = *(int**)((char*)(*(void**)(p.kernelParams)) + 8 * 4);
    int* addr2 = *(int**)((char*)(*(void**)(p.kernelParams)) + 8 * 5);
    cuda_addr_for_sm80_addrs.push_back(addr1);
    cuda_addr_for_sm80_addrs.push_back(addr2);
  }

  if (strcmp(name, "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei6__halfS1_fS1_Lb1ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT3_PS8_PKS3_SD_PKT5_S7_PSE_PvlPS3_Pi") == 0
      || strcmp(name, "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb1ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT3_PS8_PKS3_SD_PKT5_S7_PSE_PvlPS3_Pi") == 0
      || strcmp(name, "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Eif13__nv_bfloat16fS1_Lb1ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT3_PS8_PKS3_SD_PKT5_S7_PSE_PvlPS3_Pi") == 0) {
    
    int* addr1 = *(int**)(*(void**)((char*)p.kernelParams + 8 * 4));
    int* addr2 = *(int**)(*(void**)((char*)p.kernelParams + 8 * 5));
    cublasSplitKreduce_kernel_addrs.push_back(addr1);
    cublasSplitKreduce_kernel_addrs.push_back(addr2);
  }

  kernel_node_params.insert({node_idx, p.kernelParams});
}

void saveGraphMemsetNodeParams(CUDA_MEMSET_NODE_PARAMS pNodeParams) {
  SERVERLESS_LOG("memset_params.dst = %p\n", (void*)pNodeParams.dst);
  SERVERLESS_LOG("memset_params.elementSize = %d\n", pNodeParams.elementSize);
  SERVERLESS_LOG("memset_params.height = %ld\n", pNodeParams.height);
  SERVERLESS_LOG("memset_params.pitch = %ld\n", pNodeParams.pitch);
  SERVERLESS_LOG("memset_params.value = %d\n", pNodeParams.value);
  SERVERLESS_LOG("memset_params.width = %ld\n", pNodeParams.width);
}

void* getSharedObjectBaseAddress() {
    std::ifstream mapsFile("/proc/self/maps");
    if (!mapsFile) {
        std::cerr << "Failed to open /proc/self/maps" << std::endl;
        return nullptr;
    }

    std::string pattern = ".*\\/usr\\/local\\/cuda-12\\.4\\/targets\\/x86_64-linux\\/lib\\/libcublasLt\\.so\\.12\\.4\\.2\\.65.*";
    std::regex regex(pattern);
    std::smatch match;

    std::string line;
    while (std::getline(mapsFile, line)) {
      SERVERLESS_LOG("line = %s\n", line.c_str());
      if (std::regex_search(line, match, regex)) {
          std::string str12 = line.substr(0, 12);
          SERVERLESS_LOG("addressStr = %s\n", str12.c_str());
          void* baseAddress = (void*)std::stoul(str12, nullptr, 16);
          return baseAddress;
      }
    }

    std::cerr << "Shared object not found in /proc/self/maps" << std::endl;
    return nullptr;
}

void CUDAGraph::constructFunc2Lib() {
  func2lib_["_ZN2at6native44_GLOBAL__N__d3ae08d1_11_Indexing_cu_89862edb21indexSelectLargeIndexIN3c104HalfEljLi2ELi2ELin2ELb1EEEvNS_4cuda6detail10TensorInfoIT_T1_EESA_NS7_IT0_S9_EEiiS9_S9_l"] = "/home/zsx/pytorch/torch/lib/libtorch_cuda.so";
  func2lib_["_ZN4vllm15rms_norm_kernelIN3c104HalfEEEvPT_PKS3_S6_fii"] = "/home/zsx/vllm/vllm/_C.cpython-311-x86_64-linux-gnu.so";
  func2lib_["_ZN4vllm19silu_and_mul_kernelIN3c104HalfEEEvPT_PKS3_i"] = "/home/zsx/vllm/vllm/_C.cpython-311-x86_64-linux-gnu.so";
  func2lib_["_ZN4vllm23rotary_embedding_kernelIN3c104HalfELb1EEEvPKlPT_S6_PKS5_illiii"] = "/home/zsx/vllm/vllm/_C.cpython-311-x86_64-linux-gnu.so";
  func2lib_["_ZN4vllm24reshape_and_cache_kernelIttLb0EEEvPKT_S3_PT0_S5_PKliiiiii"] = "/home/zsx/vllm/vllm/_C.cpython-311-x86_64-linux-gnu.so";
  func2lib_["_ZN4vllm25fused_add_rms_norm_kernelIN3c104HalfEEEvPT_S4_PKS3_fii"] = "/home/zsx/vllm/vllm/_C.cpython-311-x86_64-linux-gnu.so";
  func2lib_["_ZN4vllm25paged_attention_v1_kernelIttLi128ELi16ELi128ELb0EEEvPT_PKS1_PKT0_S7_ifPKiS9_iPKfiii"] = "/home/zsx/vllm/vllm/_C.cpython-311-x86_64-linux-gnu.so";
  func2lib_["_ZN4vllm25paged_attention_v2_kernelIttLi128ELi16ELi128ELb0ELi512EEEvPfS1_PT_PKS2_PKT0_S8_ifPKiSA_iPKfiii"] = "/home/zsx/vllm/vllm/_C.cpython-311-x86_64-linux-gnu.so";
  func2lib_["_ZN4vllm32paged_attention_v2_reduce_kernelItLi128ELi128ELi512EEEvPT_PKfS4_PKS1_PKii"] = "/home/zsx/vllm/vllm/_C.cpython-311-x86_64-linux-gnu.so";
}

void checkGraphKernelNodeParams(
  void* &saved_ctx,
  const CUDA_KERNEL_NODE_PARAMS &pNodeParams) {
  // ctx is the same across all nodes
  if (saved_ctx == 0) {
    saved_ctx = pNodeParams.ctx;
  } else {
    if (saved_ctx != pNodeParams.ctx) {
      TORCH_CHECK(false, "ctx is different");
    }
  }

  // extra is nil across all nodes
  if (pNodeParams.extra != 0) {
    TORCH_CHECK(false, "extra is not 0");
  }

  // func should not be nullptr, and we then could set kern to nullptr
  if (pNodeParams.func == nullptr) {
    TORCH_CHECK(false, "func is nullptr");
  }
}

#define SAVE_GRAPH_BEGIN_IDX 0

void CUDAGraph::loadCublassModule(const std::vector<cudaGraphNode_t> &cublass_nodes) {
  for (auto cublass_node : cublass_nodes) {
    CUDA_KERNEL_NODE_PARAMS pNodeParams;
    CU_CALL(cuGraphKernelNodeGetParams(cublass_node, &pNodeParams));
    CUmodule cublass_mod;
    CU_CALL(cuFuncGetModule(&cublass_mod, pNodeParams.func));
    cublass_mods.push_back(cublass_mod);

    unsigned int count;
    CU_CALL(cuModuleGetFunctionCount(&count, cublass_mod));

    CUfunction *functions = new CUfunction[count];
    CU_CALL(cuModuleEnumerateFunctions(functions, count, cublass_mod));
    for (unsigned int i = 0; i < count; i++) {
      const char *func_name_in_module;
      CU_CALL(cuFuncGetName(&func_name_in_module, functions[i]));
      SERVERLESS_LOG("func_name in cublass_mod = %s\n", func_name_in_module);
    }
    delete []functions;
  }
}

void CUDAGraph::_save_cuda_graph() {
  // bind_core(0);
  size_t numCudaGraphNodes;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numCudaGraphNodes));
  SERVERLESS_LOG("cuda graph nodes: %ld\n", numCudaGraphNodes);
  std::vector<cudaGraphNode_t> all_nodes(numCudaGraphNodes);
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, all_nodes.data(), &numCudaGraphNodes));

  std::vector<CUgraphNodeParams_t> saved_nodes_params;
  std::vector<CUgraphNode> saved_nodes;
  std::map<cudaGraphNode_t, int> node_to_idx;

  size_t numKernelNodes = 0;
  size_t numMemsetNodes = 0;

  // SERVERLESS_LOG("base addr: %p\n", getSharedObjectBaseAddress());

  void *ctx = 0;
  // std::vector<cudaGraphNode_t> cublass_nodes;
  // cublass_nodes.push_back(all_nodes[1]);
  // cublass_nodes.push_back(all_nodes[3]);
  // cublass_nodes.push_back(all_nodes[4]);
  // cublass_nodes.push_back(all_nodes[5]);
  // cublass_nodes.push_back(all_nodes[6]);
  // cublass_nodes.push_back(all_nodes[8]);
  // loadCublassModule(cublass_nodes);

  SERVERLESS_LOG("============== begin_nodes ==============\n");

  for (size_t i = 0; i < numCudaGraphNodes; i++) {
    CUgraphNode node = all_nodes[i];
    node_to_idx.insert(std::make_pair(node, i));
    if (i < SAVE_GRAPH_BEGIN_IDX) {
      SERVERLESS_LOG("add nodes skip idx: %ld\n", i);
      continue;
    }

    CUgraphNodeType type;
    CU_CALL(cuGraphNodeGetType(node, &type));
    SERVERLESS_LOG("============================\n");
    SERVERLESS_LOG("node idx: %ld, node type: %d\n", i, type);

    CUgraphNodeParams_t param;
    param.node_idx = i;
    param.type = type;

    if (!load_graph_) {
      switch (type) {
        case CU_GRAPH_NODE_TYPE_KERNEL: {
          CUDA_KERNEL_NODE_PARAMS pNodeParams;
          CU_CALL(cuGraphKernelNodeGetParams(node, &pNodeParams));
          param.kernel_params = pNodeParams;
          saveGraphKernelNodeParams(pNodeParams);
          checkGraphKernelNodeParams(ctx, pNodeParams);
          numKernelNodes++;
          break;
        }
        case CU_GRAPH_NODE_TYPE_MEMSET: {
          CUDA_MEMSET_NODE_PARAMS pNodeParams;
          CU_CALL(cuGraphMemsetNodeGetParams(node, &pNodeParams));
          param.memset_params = pNodeParams;
          saveGraphMemsetNodeParams(pNodeParams);
          numMemsetNodes++;
          break;
        }
        default: {
          SERVERLESS_LOG("error node type: %d\n", type);
          break;
        }
      }
    }

    saved_nodes_params.push_back(param);
    saved_nodes.push_back(node);
  }

  SERVERLESS_LOG("============== end_nodes ==============\n");

  SERVERLESS_LOG("numKernelNodes: %ld, numMemsetNodes: %ld\n", numKernelNodes, numMemsetNodes);

  if (!load_graph_) {
    CUgraphNode all_from[MAX_NODE_AND_EDGE_NUM];
    CUgraphNode all_to[MAX_NODE_AND_EDGE_NUM];
    size_t all_savedEdges = MAX_NODE_AND_EDGE_NUM;
    CU_CALL(cuGraphGetEdges(graph_, all_from, all_to, &all_savedEdges));

    std::vector<CUgraphNode> from;
    std::vector<CUgraphNode> to;
    for (size_t i = 0; i < all_savedEdges; i++) {
      if (node_to_idx[all_from[i]] + 1 != node_to_idx[all_to[i]]) {
        TORCH_CHECK(false, "dependencies error from_idx: %d, to_idx: %d\n", node_to_idx[all_from[i]], node_to_idx[all_to[i]])
      }
      if (node_to_idx[all_from[i]] < SAVE_GRAPH_BEGIN_IDX || node_to_idx[all_to[i]] < SAVE_GRAPH_BEGIN_IDX) {
        SERVERLESS_LOG("dependencies skip from_idx: %d, to_idx: %d\n", node_to_idx[all_from[i]], node_to_idx[all_to[i]]);
        continue;
      }
      SERVERLESS_LOG("dependencies from_idx: %d, to_idx: %d\n", node_to_idx[all_from[i]], node_to_idx[all_to[i]]);
      from.push_back(all_from[i]);
      to.push_back(all_to[i]);
    }
  }

  if (load_graph_)
    load_kernel_graph(saved_nodes);
}

// =================================== load cuda graph API begin ===================================

void CUDAGraph::loadGraphKernelNodeFunc_fromCuBlass(const char *func_name, CUfunction &f) {
  bool found = false;

  for (auto cublass_mod : cublass_mods) {
    unsigned int count;
    CU_CALL(cuModuleGetFunctionCount(&count, cublass_mod));

    CUfunction *functions = new CUfunction[count];
    CU_CALL(cuModuleEnumerateFunctions(functions, count, cublass_mod));
    for (unsigned int i = 0; i < count; i++) {
      const char *func_name_in_module;
      CU_CALL(cuFuncGetName(&func_name_in_module, functions[i]));
      // SERVERLESS_LOG("func_name in cublass_mod = %s\n", func_name_in_module);
      if (strcmp(func_name_in_module, func_name) == 0) {
        CU_CALL(cuModuleGetFunction(&f, cublass_mod, func_name));
        found = true;
      }
    }
    delete []functions;
  }

  if (!found) {
    SERVERLESS_LOG("loadGraphKernelNodeFunc_fromCuBlass: !!! NOT FOUND !!! func_name: %s\n", func_name);
    TORCH_CHECK(false);
  }
}

void CUDAGraph::loadGraphKernelNodeFunc_fromSO(const char *func_name, CUfunction &f, const std::string &libraryName) {
  // Load the library
  void* libraryHandle = dlopen(libraryName.c_str(), RTLD_LAZY);
  if (!libraryHandle) {
    SERVERLESS_LOG("loadGraphKernelNodeFunc_fromSO: Failed to load library %s\n", libraryName.c_str());
    TORCH_CHECK(false);
  }

  // Load the function
  void* functionHandle = dlsym(libraryHandle, func_name);
  if (!functionHandle) {
    dlclose(libraryHandle);
    SERVERLESS_LOG("loadGraphKernelNodeFunc_fromSO: Failed to load function %s\n", func_name);
    TORCH_CHECK(false);
  }

  cudaFunction_t functionPtr;
  AT_CUDA_CHECK(cudaGetFuncBySymbol(&functionPtr, functionHandle));

  f = functionPtr;
}

void CUDAGraph::loadGraphKernelNodeFunc(const char *func_name, CUfunction &f) {
  auto it1 = func2ptr_.find(func_name);
  if (it1 != func2ptr_.end()) {
    // load func from saved func->ptr
    f = it1->second;
    return;
  }

  auto it2 = func2lib_.find(func_name);
  if (it2 != func2lib_.end()) {
    // load func from .so files
    std::string &libraryName = it2->second;
    loadGraphKernelNodeFunc_fromSO(func_name, f, libraryName);
    return;
  }

  // load func from cublass module
  loadGraphKernelNodeFunc_fromCuBlass(func_name, f);
  return;
}

bool isCublassFunc(const char *func_name) {
  return 
    strcmp(func_name, "ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_256x64_ldg8_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_64x128_ldg8_f2f_stages_64x4_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_tn") == 0
|| strcmp(func_name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x128x32_stage4_warpsize2x2x1_tensor16x8x16_kernel") == 0
    || strcmp(func_name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize192x128x32_stage3_warpsize4x2x1_tensor16x8x16_kernel") == 0
    || strcmp(func_name, "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei6__halfS1_fS1_Lb1ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT3_PS8_PKS3_SD_PKT5_S7_PSE_PvlPS3_Pi") == 0
    || strcmp(func_name, "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb1ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT3_PS8_PKS3_SD_PKT5_S7_PSE_PvlPS3_Pi") == 0
    || strcmp(func_name, "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Eif13__nv_bfloat16fS1_Lb1ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT3_PS8_PKS3_SD_PKT5_S7_PSE_PvlPS3_Pi") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_s16816gemm_bf16_128x64_ldg8_stages_64x4_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x128_ldg8_f2f_stages_64x4_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x5_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_64x64_ldg8_relu_f2f_stages_64x5_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x5_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x64_ldg8_relu_f2f_stages_64x5_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_32x6_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x4_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_256x64_ldg8_relu_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn") == 0
    || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn") == 0
    || strcmp(func_name, "_ZN7cutlass7Kernel2I66cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x6_tn_align8EEvNT_6ParamsE") == 0
    || strcmp(func_name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x96x64_stage4_warpsize2x2x1_tensor16x8x16_kernel") == 0
    || strcmp(func_name, "_ZN7cutlass7Kernel2I65cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x6_tn_align8EEvNT_6ParamsE") == 0
    || strcmp(func_name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x64x64_stage4_warpsize2x2x1_tensor16x8x16_kernel") == 0
    || strcmp(func_name, "_ZN7cutlass7Kernel2I66cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8EEvNT_6ParamsE") == 0
    || strcmp(func_name, "_ZN7cutlass7Kernel2I56cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8EEvNT_6ParamsE") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn") == 0
    || strcmp(func_name, "_ZN7cutlass7Kernel2I67cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x6_tn_align8EEvNT_6ParamsE") == 0
    || strcmp(func_name, "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE") == 0
    || strcmp(func_name, "_ZN8internal5gemvx6kernelIii13__nv_bfloat16S2_S2_fLb0ELb1ELb1ELb0ELi7ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS2_ES6_S4_IS2_EfEEENSt9enable_ifIXntT5_EvE4typeET11_") == 0
    || strcmp(func_name, "_Z17gemv2T_kernel_valIii13__nv_bfloat16S0_S0_fLi128ELi16ELi4ELi4ELb0ELb1E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES4_S2_IS0_EfEEvT11_T4_S8_") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn") == 0
    || strcmp(func_name, "_ZN7cutlass7Kernel2I63cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE") == 0
    || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x6_tn") == 0
    || strcmp(func_name, "_ZN7cutlass7Kernel2I57cutlass_80_tensorop_s16816gemm_bf16_64x128_64x4_tn_align8EEvNT_6ParamsE") == 0
    || strcmp(func_name, "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE") == 0
    || strcmp(func_name, "_Z17gemv2T_kernel_valIii13__nv_bfloat16fffLi128ELi16ELi4ELi4ELb0ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES2_IKfES2_IfEfEEvT11_T4_SA_") == 0;
}

void CUDAGraph::loadGraphKernelNodeFuncParams_ampere(void** &params, int node_idx) {
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  uint64_t *params_start_addr = (uint64_t*)(*(void**)((uint64_t*)params + 0));
  // first input matrix addr
  *params_start_addr = (*malloc_addrs)[malloc_idxs[0]];
  // second input matrix addr
  *(params_start_addr + 1) = (*malloc_addrs)[malloc_idxs[1]];
  // output matrix addr
  *(params_start_addr + 2) = (*malloc_addrs)[malloc_idxs[2]];
  // tmp matrix addr
  // maybe not an addr
  if (malloc_idxs[3] != -1)
    *(params_start_addr + 15) = (*malloc_addrs)[malloc_idxs[3]];
  // output matrix addr again
  *(params_start_addr + 21) = (*malloc_addrs)[malloc_idxs[2]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_ampere_208(void** &params, int node_idx) {
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  uint64_t *params_start_addr = (uint64_t*)(*(void**)((uint64_t*)params + 0));
  *params_start_addr = (*malloc_addrs)[malloc_idxs[0]];
  *(params_start_addr + 1) = (*malloc_addrs)[malloc_idxs[1]];
  *(params_start_addr + 2) = (*malloc_addrs)[malloc_idxs[2]];
  *(params_start_addr + 15) = (*malloc_addrs)[malloc_idxs[3]];
  *(params_start_addr + 22) = (*malloc_addrs)[malloc_idxs[4]];
  *(params_start_addr + 23) = (*malloc_addrs)[malloc_idxs[2]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_sm80(void** &params, int node_idx) {
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  uint64_t *params_start_addr = (uint64_t*)(*(void**)((uint64_t*)params + 0));
  // first matrix addr
  *params_start_addr = (*malloc_addrs)[malloc_idxs[0]];
  // second matrix addr
  *(params_start_addr + 1) = (*malloc_addrs)[malloc_idxs[1]];
  // third matrix addr
  *(params_start_addr + 2) = (*malloc_addrs)[malloc_idxs[2]];
  // fourth matrix addr
  *(params_start_addr + 6) = (*malloc_addrs)[malloc_idxs[3]];

  *(params_start_addr + 4) = (uint64_t)cuda_addr_for_sm80_addrs[cuda_addr_for_sm80_idx];
  cuda_addr_for_sm80_idx = (cuda_addr_for_sm80_idx + 1) % cuda_addr_for_sm80_addrs.size();
  *(params_start_addr + 5) = (uint64_t)cuda_addr_for_sm80_addrs[cuda_addr_for_sm80_idx];
   cuda_addr_for_sm80_idx = (cuda_addr_for_sm80_idx + 1) % cuda_addr_for_sm80_addrs.size();
}

void CUDAGraph::loadGraphKernelNodeFuncParams_cublasSplitKreduce_kernel(void** &params, int node_idx) {
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  *(uint64_t*)(*(void**)((uint64_t*)params + 1)) = (*malloc_addrs)[malloc_idxs[0]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 2)) = (*malloc_addrs)[malloc_idxs[1]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 3)) = (*malloc_addrs)[malloc_idxs[2]];

  *(uint64_t*)(*(void**)((uint64_t*)params + 4)) = (uint64_t)cublasSplitKreduce_kernel_addrs[cublasSplitKreduce_kernel_idx];
  cublasSplitKreduce_kernel_idx = (cublasSplitKreduce_kernel_idx + 1) % cublasSplitKreduce_kernel_addrs.size();
  *(uint64_t*)(*(void**)((uint64_t*)params + 5)) = (uint64_t)cublasSplitKreduce_kernel_addrs[cublasSplitKreduce_kernel_idx];
  cublasSplitKreduce_kernel_idx = (cublasSplitKreduce_kernel_idx + 1) % cublasSplitKreduce_kernel_addrs.size();
}

void CUDAGraph::loadGraphKernelNodeFuncParams_cutlass_80_tensorop(void** &params, int node_idx) {
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  uint64_t *params_start_addr = (uint64_t*)(*(void**)((uint64_t*)params + 0));
  *(params_start_addr + 43) = (*malloc_addrs)[malloc_idxs[0]];
  *(params_start_addr + 44) = (*malloc_addrs)[malloc_idxs[1]];
  *(params_start_addr + 45) = (*malloc_addrs)[malloc_idxs[2]];
  *(params_start_addr + 46) = (*malloc_addrs)[malloc_idxs[3]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_cutlass_80_tensorop_360(void** &params, int node_idx) {
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  uint64_t *params_start_addr = (uint64_t*)(*(void**)((uint64_t*)params + 0));
  *(params_start_addr + 35) = (*malloc_addrs)[malloc_idxs[0]];
  *(params_start_addr + 36) = (*malloc_addrs)[malloc_idxs[1]];
  *(params_start_addr + 37) = (*malloc_addrs)[malloc_idxs[2]];
  *(params_start_addr + 38) = (*malloc_addrs)[malloc_idxs[3]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_cublasGemv(void** &params, int node_idx) {
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  uint64_t *params_start_addr = (uint64_t*)(*(void**)((uint64_t*)params + 0));
  *(params_start_addr) = (*malloc_addrs)[malloc_idxs[0]];
  *(params_start_addr + 2) = (*malloc_addrs)[malloc_idxs[1]];
  *(params_start_addr + 4) = (*malloc_addrs)[malloc_idxs[2]];
  *(params_start_addr + 6) = (*malloc_addrs)[malloc_idxs[3]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_cublasGemv_152(void** &params, int node_idx) {
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  uint64_t *params_start_addr = (uint64_t*)(*(void**)((uint64_t*)params + 0));
  *(params_start_addr) = (*malloc_addrs)[malloc_idxs[0]];
  *(params_start_addr + 2) = (*malloc_addrs)[malloc_idxs[1]];
  *(params_start_addr + 4) = (*malloc_addrs)[malloc_idxs[2]];
  *(params_start_addr + 6) = (*malloc_addrs)[malloc_idxs[3]];
  *(params_start_addr + 12) = (*malloc_addrs)[malloc_idxs[4]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_cublasGemv_152_2(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  uint64_t *params_start_addr = (uint64_t*)(*(void**)((uint64_t*)params + 0));
  *params_start_addr = (*malloc_addrs)[malloc_idxs[0]];
  *(params_start_addr + 2) = (*malloc_addrs)[malloc_idxs[1]];
  *(params_start_addr + 6) = (*malloc_addrs)[malloc_idxs[2]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_fromCuBlass(const char *func_name, void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  if (strcmp(func_name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x128x32_stage4_warpsize2x2x1_tensor16x8x16_kernel") == 0
    || strcmp(func_name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize192x128x32_stage3_warpsize4x2x1_tensor16x8x16_kernel") == 0
    || strcmp(func_name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x96x64_stage4_warpsize2x2x1_tensor16x8x16_kernel") == 0
    || strcmp(func_name, "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x64x64_stage4_warpsize2x2x1_tensor16x8x16_kernel") == 0) {
    loadGraphKernelNodeFuncParams_sm80(params, node_idx);
  } else if (strcmp(func_name, "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei6__halfS1_fS1_Lb1ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT3_PS8_PKS3_SD_PKT5_S7_PSE_PvlPS3_Pi") == 0
  || strcmp(func_name, "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb1ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT3_PS8_PKS3_SD_PKT5_S7_PSE_PvlPS3_Pi") == 0
  || strcmp(func_name, "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Eif13__nv_bfloat16fS1_Lb1ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT3_PS8_PKS3_SD_PKT5_S7_PSE_PvlPS3_Pi") == 0) {
    loadGraphKernelNodeFuncParams_cublasSplitKreduce_kernel(params, node_idx);
  } else if (strcmp(func_name, "ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x5_tn") == 0
          || strcmp(func_name, "ampere_fp16_s16816gemm_fp16_64x64_ldg8_relu_f2f_stages_64x5_tn") == 0
          || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x5_tn") == 0
          || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x64_ldg8_relu_f2f_stages_64x5_tn") == 0
          || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_32x6_tn") == 0
          || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x4_tn") == 0
          || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_256x64_ldg8_relu_f2f_stages_64x3_tn") == 0
          || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_64x3_tn") == 0
          || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn") == 0
          || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn") == 0
          || strcmp(func_name, "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x6_tn") == 0) {
    loadGraphKernelNodeFuncParams_ampere_208(params, node_idx);
  } else if (strcmp(func_name, "_ZN7cutlass7Kernel2I66cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x6_tn_align8EEvNT_6ParamsE") == 0
          || strcmp(func_name, "_ZN7cutlass7Kernel2I65cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x6_tn_align8EEvNT_6ParamsE") == 0
          || strcmp(func_name, "_ZN7cutlass7Kernel2I66cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8EEvNT_6ParamsE") == 0
          || strcmp(func_name, "_ZN7cutlass7Kernel2I67cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x6_tn_align8EEvNT_6ParamsE") == 0) {
    loadGraphKernelNodeFuncParams_cutlass_80_tensorop(params, node_idx);
  } else if (strcmp(func_name, "_ZN7cutlass7Kernel2I56cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8EEvNT_6ParamsE") == 0
          || strcmp(func_name, "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE") == 0
          || strcmp(func_name, "_ZN7cutlass7Kernel2I63cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE") == 0
          || strcmp(func_name, "_ZN7cutlass7Kernel2I57cutlass_80_tensorop_s16816gemm_bf16_64x128_64x4_tn_align8EEvNT_6ParamsE") == 0
          || strcmp(func_name, "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE") == 0) {
    loadGraphKernelNodeFuncParams_cutlass_80_tensorop_360(params, node_idx);
  } else if (strcmp(func_name, "_ZN8internal5gemvx6kernelIii13__nv_bfloat16S2_S2_fLb0ELb1ELb1ELb0ELi7ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS2_ES6_S4_IS2_EfEEENSt9enable_ifIXntT5_EvE4typeET11_") == 0) {
    loadGraphKernelNodeFuncParams_cublasGemv(params, node_idx);
  } else if (strcmp(func_name, "_Z17gemv2T_kernel_valIii13__nv_bfloat16S0_S0_fLi128ELi16ELi4ELi4ELb0ELb1E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES4_S2_IS0_EfEEvT11_T4_S8_") == 0) {
    loadGraphKernelNodeFuncParams_cublasGemv_152(params, node_idx);
  } else if (strcmp(func_name, "_Z17gemv2T_kernel_valIii13__nv_bfloat16fffLi128ELi16ELi4ELi4ELb0ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES2_IKfES2_IfEfEEvT11_T4_SA_") == 0) {
    loadGraphKernelNodeFuncParams_cublasGemv_152_2(params, node_idx);
  }
  else {
    loadGraphKernelNodeFuncParams_ampere(params, node_idx);
  }
}

uint64_t CUDAGraph::get_key_addr(uint64_t addr) {
  return addr + key_offset;
}

uint64_t CUDAGraph::get_value_addr(uint64_t addr) {
  return addr + value_offset;
}

void CUDAGraph::loadGraphKernelNodeFuncParams_static_load(void** &params, int node_idx) {
  if (node_params.find(node_idx) == node_params.end()) {
    TORCH_CHECK(false, "node ", node_idx, " not exists");
  }
  CUgraphNodeParams *node_param = node_params[node_idx];


  if (node_param->corresponding_layer_one_idx != -1) {
    params = kernel_node_params[node_param->corresponding_layer_one_idx];
  } else {
    params = (void**)malloc(sizeof(void*) * node_param->other_kernel_params.paramPtrs.size());
    for (uint64_t i = 0; i < node_param->other_kernel_params.paramSizes.size(); i++) {
      *(void**)((uint64_t*)params + i) = malloc(node_param->other_kernel_params.paramSizes[i]);
      memcpy(*(void**)((uint64_t*)params + i), node_param->other_kernel_params.paramPtrs[i], node_param->other_kernel_params.paramSizes[i]);
    }
    kernel_node_params.insert({node_idx, params});
  }
}

void CUDAGraph::loadGraphKernelNodeFuncParams_paged_attention_v1_kernel(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  *(uint64_t*)(*(void**)((uint64_t*)params + 0)) = (*malloc_addrs)[malloc_idxs[0]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 1)) = (*malloc_addrs)[malloc_idxs[1]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 2)) = (*malloc_addrs)[malloc_idxs[2]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 3)) = (*malloc_addrs)[malloc_idxs[3]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 6)) = (*malloc_addrs)[malloc_idxs[4]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 7)) = (*malloc_addrs)[malloc_idxs[5]];
  if (malloc_idxs[6] != -1)
    *(uint64_t*)(*(void**)((uint64_t*)params + 9)) = (*malloc_addrs)[malloc_idxs[6]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_paged_attention_v2_kernel(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  *(uint64_t*)(*(void**)((uint64_t*)params + 0)) = (*malloc_addrs)[malloc_idxs[0]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 1)) = (*malloc_addrs)[malloc_idxs[1]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 2)) = (*malloc_addrs)[malloc_idxs[2]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 3)) = (*malloc_addrs)[malloc_idxs[3]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 4)) = (*malloc_addrs)[malloc_idxs[4]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 5)) = (*malloc_addrs)[malloc_idxs[5]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 8)) = (*malloc_addrs)[malloc_idxs[6]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 9)) = (*malloc_addrs)[malloc_idxs[7]];
  if (malloc_idxs[8] != -1) {
    *(uint64_t*)(*(void**)((uint64_t*)params + 11)) = (*malloc_addrs)[malloc_idxs[8]];
  }
}

void CUDAGraph::loadGraphKernelNodeFuncParams_paged_attention_v2_reduce_kernel(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  *(uint64_t*)(*(void**)((uint64_t*)params + 0)) = (*malloc_addrs)[malloc_idxs[0]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 1)) = (*malloc_addrs)[malloc_idxs[1]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 2)) = (*malloc_addrs)[malloc_idxs[2]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 3)) = (*malloc_addrs)[malloc_idxs[3]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 4)) = (*malloc_addrs)[malloc_idxs[4]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_fused_add_rms_norm_kernel(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  *(uint64_t*)(*(void**)((uint64_t*)params + 0)) = (*malloc_addrs)[malloc_idxs[0]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 1)) = (*malloc_addrs)[malloc_idxs[1]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 2)) = (*malloc_addrs)[malloc_idxs[2]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_reshape_and_cache_kernel(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  *(uint64_t*)(*(void**)((uint64_t*)params + 0)) = get_key_addr((*malloc_addrs)[malloc_idxs[0]]);
  *(uint64_t*)(*(void**)((uint64_t*)params + 1)) = get_value_addr((*malloc_addrs)[malloc_idxs[1]]);
  *(uint64_t*)(*(void**)((uint64_t*)params + 2)) = (*malloc_addrs)[malloc_idxs[2]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 3)) = (*malloc_addrs)[malloc_idxs[3]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 4)) = (*malloc_addrs)[malloc_idxs[4]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_rotary_embedding_kernel(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  *(uint64_t*)(*(void**)((uint64_t*)params + 0)) = (*malloc_addrs)[malloc_idxs[0]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 1)) = (*malloc_addrs)[malloc_idxs[1]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 2)) = get_key_addr((*malloc_addrs)[malloc_idxs[2]]);
  *(uint64_t*)(*(void**)((uint64_t*)params + 3)) = (*malloc_addrs)[malloc_idxs[3]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_silu_and_mul_kernel(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  *(uint64_t*)(*(void**)((uint64_t*)params + 0)) = (*malloc_addrs)[malloc_idxs[0]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 1)) = (*malloc_addrs)[malloc_idxs[1]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_rms_norm_kernel(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  *(uint64_t*)(*(void**)((uint64_t*)params + 0)) = (*malloc_addrs)[malloc_idxs[0]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 1)) = (*malloc_addrs)[malloc_idxs[1]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 2)) = (*malloc_addrs)[malloc_idxs[2]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_layer_norm_kernel(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  *(uint64_t*)(*(void**)((uint64_t*)params + 2)) = (*malloc_addrs)[malloc_idxs[0]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 3)) = (*malloc_addrs)[malloc_idxs[1]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 4)) = (*malloc_addrs)[malloc_idxs[2]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 5)) = (*malloc_addrs)[malloc_idxs[3]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 6)) = (*malloc_addrs)[malloc_idxs[4]];
  *(uint64_t*)(*(void**)((uint64_t*)params + 7)) = (*malloc_addrs)[malloc_idxs[5]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_vectorized_elementwise_kernel_TensorIteratorBase(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  uint64_t *params_start_addr = (uint64_t*)(*(void**)((uint64_t*)params + 2));
  *params_start_addr = (*malloc_addrs)[malloc_idxs[0]];
  *(params_start_addr + 1) = (*malloc_addrs)[malloc_idxs[1]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_vectorized_elementwise_kernel(void** &params, int node_idx) {
  loadGraphKernelNodeFuncParams_static_load(params, node_idx);
  CUgraphNodeParams *node_param = node_params[node_idx];
  const std::vector<int> &malloc_idxs = node_param->other_kernel_params.malloc_idxs;

  uint64_t *params_start_addr = (uint64_t*)(*(void**)((uint64_t*)params + 2));
  *params_start_addr = (*malloc_addrs)[malloc_idxs[0]];
  *(params_start_addr + 1) = (*malloc_addrs)[malloc_idxs[1]];
  *(params_start_addr + 2) = (*malloc_addrs)[malloc_idxs[2]];
}

void CUDAGraph::loadGraphKernelNodeFuncParams_fromOtherKernel(const char *func_name, void** &params, int node_idx) {
  if (strcmp(func_name, "_ZN4vllm25paged_attention_v1_kernelIttLi128ELi16ELi128ELb0EEEvPT_PKS1_PKT0_S7_ifPKiS9_iPKfiii") == 0
    || strcmp(func_name, "_ZN4vllm25paged_attention_v1_kernelI13__nv_bfloat16S1_Li128ELi16ELi128ELb0EEEvPT_PKS2_PKT0_S8_ifPKiSA_iPKfiii") == 0
    || strcmp(func_name, "_ZN4vllm25paged_attention_v1_kernelI13__nv_bfloat16S1_Li64ELi16ELi128ELb0EEEvPT_PKS2_PKT0_S8_ifPKiSA_iPKfiii") == 0) {
    loadGraphKernelNodeFuncParams_paged_attention_v1_kernel(params, node_idx);
  }
  else if (strcmp(func_name, "_ZN4vllm25paged_attention_v2_kernelIttLi128ELi16ELi128ELb0ELi512EEEvPfS1_PT_PKS2_PKT0_S8_ifPKiSA_iPKfiii") == 0
        || strcmp(func_name, "_ZN4vllm25paged_attention_v2_kernelI13__nv_bfloat16S1_Li128ELi16ELi128ELb0ELi512EEEvPfS2_PT_PKS3_PKT0_S9_ifPKiSB_iPKfiii") == 0
        || strcmp(func_name, "_ZN4vllm25paged_attention_v2_kernelI13__nv_bfloat16S1_Li64ELi16ELi128ELb0ELi512EEEvPfS2_PT_PKS3_PKT0_S9_ifPKiSB_iPKfiii") == 0) {
    loadGraphKernelNodeFuncParams_paged_attention_v2_kernel(params, node_idx);
  }
  else if (strcmp(func_name, "_ZN4vllm32paged_attention_v2_reduce_kernelItLi128ELi128ELi512EEEvPT_PKfS4_PKS1_PKii") == 0
        || strcmp(func_name, "_ZN4vllm32paged_attention_v2_reduce_kernelI13__nv_bfloat16Li128ELi128ELi512EEEvPT_PKfS5_PKS2_PKii") == 0
        || strcmp(func_name, "_ZN4vllm32paged_attention_v2_reduce_kernelI13__nv_bfloat16Li64ELi128ELi512EEEvPT_PKfS5_PKS2_PKii") == 0) {
    loadGraphKernelNodeFuncParams_paged_attention_v2_reduce_kernel(params, node_idx);
  }
  else if (strcmp(func_name, "_ZN4vllm25fused_add_rms_norm_kernelIN3c104HalfEEEvPT_S4_PKS3_fii") == 0
        || strcmp(func_name, "_ZN4vllm25fused_add_rms_norm_kernelIN3c108BFloat16EEEvPT_S4_PKS3_fii") == 0) {
    loadGraphKernelNodeFuncParams_fused_add_rms_norm_kernel(params, node_idx);
  }
  else if (strcmp(func_name, "_ZN4vllm24reshape_and_cache_kernelIttLb0EEEvPKT_S3_PT0_S5_PKliiiiii") == 0
        || strcmp(func_name, "_ZN4vllm24reshape_and_cache_kernelI13__nv_bfloat16S1_Lb0EEEvPKT_S4_PT0_S6_PKliiiiii") == 0) {
    loadGraphKernelNodeFuncParams_reshape_and_cache_kernel(params, node_idx);
  }
  else if (strcmp(func_name, "_ZN4vllm23rotary_embedding_kernelIN3c104HalfELb1EEEvPKlPT_S6_PKS5_illiii") == 0
        || strcmp(func_name, "_ZN4vllm23rotary_embedding_kernelIN3c108BFloat16ELb1EEEvPKlPT_S6_PKS5_illiii") == 0) {
    loadGraphKernelNodeFuncParams_rotary_embedding_kernel(params, node_idx);
  }
  else if (strcmp(func_name, "_ZN4vllm19silu_and_mul_kernelIN3c104HalfEEEvPT_PKS3_i") == 0
        || strcmp(func_name, "_ZN4vllm19silu_and_mul_kernelIN3c108BFloat16EEEvPT_PKS3_i") == 0) {
    loadGraphKernelNodeFuncParams_silu_and_mul_kernel(params, node_idx);
  }
  else if (strcmp(func_name, "_ZN4vllm15rms_norm_kernelIN3c104HalfEEEvPT_PKS3_S6_fii") == 0
        || strcmp(func_name, "_ZN4vllm15rms_norm_kernelIN3c108BFloat16EEEvPT_PKS3_S6_fii") == 0) {
    loadGraphKernelNodeFuncParams_rms_norm_kernel(params, node_idx);
  }
  else if (strcmp(func_name, "_ZN2at6native53_GLOBAL__N__1aa0d23f_20_layer_norm_kernel_cu_9c5ada8a28vectorized_layer_norm_kernelIN3c104HalfEfEEviT0_PKT_S8_S8_PS5_S9_PS6_") == 0
        || strcmp(func_name, "_ZN2at6native53_GLOBAL__N__1aa0d23f_20_layer_norm_kernel_cu_9c5ada8a28vectorized_layer_norm_kernelIN3c108BFloat16EfEEviT0_PKT_S8_S8_PS5_S9_PS6_") == 0
        || strcmp(func_name, "_ZN2at6native53_GLOBAL__N__7ae05bd5_20_layer_norm_kernel_cu_9c5ada8a28vectorized_layer_norm_kernelIN3c108BFloat16EfEEviT0_PKT_S8_S8_PS5_S9_PS6_") == 0) {
    loadGraphKernelNodeFuncParams_layer_norm_kernel(params, node_idx);
  }
  else if (strcmp(func_name, "_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_18GeluCUDAKernelImplERNS_18TensorIteratorBaseENS0_8GeluTypeEENKUlvE0_clEvENKUlvE1_clEvEUlN3c104HalfEE_NS_6detail5ArrayIPcLi2EEEEEviT0_T1_") == 0
        || strcmp(func_name, "_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_18GeluCUDAKernelImplERNS_18TensorIteratorBaseENS0_8GeluTypeEENKUlvE0_clEvENKUlvE2_clEvEUlN3c108BFloat16EE_NS_6detail5ArrayIPcLi2EEEEEviT0_T1_") == 0) {
    loadGraphKernelNodeFuncParams_vectorized_elementwise_kernel_TensorIteratorBase(params, node_idx);
  }
  else if (strcmp(func_name, "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIN3c104HalfEEENS_6detail5ArrayIPcLi3EEEEEviT0_T1_") == 0
        || strcmp(func_name, "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIN3c108BFloat16EEENS_6detail5ArrayIPcLi3EEEEEviT0_T1_") == 0) {
    loadGraphKernelNodeFuncParams_vectorized_elementwise_kernel(params, node_idx);
  }
  else {
    SERVERLESS_LOG("not support!, func_name %s\n", func_name);
    TORCH_CHECK(false);
  }
}

void CUDAGraph::loadGraphKernelNodeFuncParams(const char *func_name, void** &params, int node_idx) {
  if (isCublassFunc(func_name)) {
    loadGraphKernelNodeFuncParams_fromCuBlass(func_name, params, node_idx);
  } else {
    loadGraphKernelNodeFuncParams_fromOtherKernel(func_name, params, node_idx);
  }
}

void CUDAGraph::loadGraphKernelNodeParams(
  const CUcontext &ctx,
  CUDA_KERNEL_NODE_PARAMS &p,
  int node_idx) {

  CUgraphNodeParams *node_param = node_params[node_idx];
  p.blockDimX = node_param->other_kernel_params.blockDimX;
  p.blockDimY = node_param->other_kernel_params.blockDimY;
  p.blockDimZ = node_param->other_kernel_params.blockDimZ;
  p.ctx = ctx;
  p.extra = nullptr;
  p.func = 0;
  p.gridDimX = node_param->other_kernel_params.gridDimX;
  p.gridDimY = node_param->other_kernel_params.gridDimY;
  p.gridDimZ = node_param->other_kernel_params.gridDimZ;
  p.kern = 0;
  p.kernelParams = 0;
  p.sharedMemBytes = node_param->other_kernel_params.sharedMemBytes;

  loadGraphKernelNodeFunc(node_param->kernel_func_name.c_str(), p.func);
  loadGraphKernelNodeFuncParams(node_param->kernel_func_name.c_str(), p.kernelParams, node_idx);
}

void CUDAGraph::loadGraphMemsetNodeParams(CUDA_MEMSET_NODE_PARAMS &p, int node_idx) {
  CUgraphNodeParams *node_param = node_params[node_idx];

  TORCH_CHECK(node_param->memset_params.dst_malloc_idx != -1, "node_param->memset_params.dst_malloc_idx is -1");
  p.dst = (*malloc_addrs)[node_param->memset_params.dst_malloc_idx];
  p.elementSize = node_param->memset_params.elementSize;
  p.height = node_param->memset_params.height;
  p.pitch = node_param->memset_params.pitch;
  p.value = node_param->memset_params.value;
  p.width = node_param->memset_params.width;
}

void CUDAGraph::read_func_params() {
  CUgraphNodeParams* node_param;

  if (func_params_file_stream.is_open()) {
    std::string line;
    while (std::getline(func_params_file_stream, line)) {
      {
        if (line.find("node idx:") != std::string::npos) {
            size_t idx = line.find(":") + 1;
            int node_idx = std::stoi(line.substr(idx + 1));
            if (node_params.find(node_idx) == node_params.end()) {
              TORCH_CHECK(false, "node_idx ", node_idx, " not exists");
            }
            node_param = node_params.find(node_idx)->second;
        }
      }

      {
        std::smatch match;
        if (line.find("addr_corresponding_malloc_idx") != std::string::npos) {
            if (node_param->node_type == 0) {
              size_t start_idx = line.find(":");
              while(1) {
                line = line.substr(start_idx + 1);
                if (line.empty()) {
                  break;
                }

                size_t end_idx = line.find(",");
                int num = std::stoi(line);
                node_param->other_kernel_params.malloc_idxs.push_back(num);
                start_idx = end_idx;
              }
            } else if (node_param->node_type == 2) {
              size_t start_idx = line.find(":");
              line = line.substr(start_idx + 1);
              int num = std::stoi(line);
              node_param->memset_params.dst_malloc_idx = num;
            } else {
              TORCH_CHECK(false, "node type not found");
            }
        }
      }
    }

    func_params_file_stream.close();
  } else {
    SERVERLESS_LOG("open file failed\n");
  }
}

CUgraphNodeParams* CUDAGraph::read_one_node(int layer_no, std::vector<CUgraphNodeParams*> &one_layer_nodes, size_t &one_layer_nodes_idx) {
  CUgraphNodeParams* node_param;
  std::string line;
  while (std::getline(node_file_stream, line)) {
    {
      if (line.rfind("node idx:", 0) == 0) {
          size_t idx = line.find(":") + 1;
          int node_idx = std::stoi(line.substr(idx + 1));
          node_param = new CUgraphNodeParams();
          node_param->node_idx = node_idx;
          node_param->corresponding_layer_one_idx = -1;

          line = line.substr(idx);
          idx = line.find(":");
          line = line.substr(idx + 1);

          int node_type = std::stoi(line);
          node_param->node_type = node_type;

          if (node_type == 2) {
            if (layer_no > 1) {
              CUgraphNodeParams* node = one_layer_nodes[one_layer_nodes_idx];
              one_layer_nodes_idx = (one_layer_nodes_idx + 1) % one_layer_nodes.size();
              if (node->node_type == 2) {
                node_param->memset_params = node->memset_params;
                return node_param;
              } else {
                SERVERLESS_LOG("node type not match, current node_idx: %d, layer one node_idx: %d\n", node_param->node_idx, node->node_idx);
              }
            }

            std::getline(node_file_stream, line);

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int elementSize = std::stoi(line.substr(idx + 1));
              node_param->memset_params.elementSize = elementSize;
            }

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int height = std::stoi(line.substr(idx + 1));
              node_param->memset_params.height = height;
            }

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int pitch = std::stoi(line.substr(idx + 1));
              node_param->memset_params.pitch = pitch;
            }

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int value = std::stoi(line.substr(idx + 1));
              node_param->memset_params.value = value;
            }

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int width = std::stoi(line.substr(idx + 1));
              node_param->memset_params.width = width;
            }

            if (layer_no == 1) {
              one_layer_nodes.push_back(node_param);
            }
            return node_param;
          } else if (node_type == 0) {
            // kernal name
            std::getline(node_file_stream, line);
            {        
              if (line.rfind("kernel_func.name mangled =", 0) == 0) {
                size_t idx = line.find("=") + 1;
                node_param->kernel_func_name = line.substr(idx + 1);
              }
            }

            if (layer_no > 1) {
              CUgraphNodeParams* node = one_layer_nodes[one_layer_nodes_idx];
              one_layer_nodes_idx = (one_layer_nodes_idx + 1) % one_layer_nodes.size();
              if (node->node_type == 0 && node->kernel_func_name == node_param->kernel_func_name) {
                node_param->other_kernel_params.blockDimX = node->other_kernel_params.blockDimX;
                node_param->other_kernel_params.blockDimY = node->other_kernel_params.blockDimY;
                node_param->other_kernel_params.blockDimZ = node->other_kernel_params.blockDimZ;
                node_param->other_kernel_params.gridDimX = node->other_kernel_params.gridDimX;
                node_param->other_kernel_params.gridDimY = node->other_kernel_params.gridDimY;
                node_param->other_kernel_params.gridDimZ = node->other_kernel_params.gridDimZ;
                node_param->other_kernel_params.sharedMemBytes = node->other_kernel_params.sharedMemBytes;

                node_param->other_kernel_params.paramSizes = node->other_kernel_params.paramSizes;
                node_param->other_kernel_params.paramPtrs = node->other_kernel_params.paramPtrs;

                node_param->corresponding_layer_one_idx = node->node_idx;

                return node_param;
              } else {
                SERVERLESS_LOG("node type not match, current node_idx: %d, layer one node_idx: %d\n", node_param->node_idx, node->node_idx);
              }
            }

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int blockDimX = std::stoi(line.substr(idx + 1));
              node_param->other_kernel_params.blockDimX = blockDimX;
            }

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int blockDimY = std::stoi(line.substr(idx + 1));
              node_param->other_kernel_params.blockDimY = blockDimY;
            }

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int blockDimZ = std::stoi(line.substr(idx + 1));
              node_param->other_kernel_params.blockDimZ = blockDimZ;
            }

            // ctx
            std::getline(node_file_stream, line);
            // extra
            std::getline(node_file_stream, line);
            // func
            std::getline(node_file_stream, line);

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int gridDimX = std::stoi(line.substr(idx + 1));
              node_param->other_kernel_params.gridDimX = gridDimX;
            }

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int gridDimY = std::stoi(line.substr(idx + 1));
              node_param->other_kernel_params.gridDimY = gridDimY;
            }

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int gridDimZ = std::stoi(line.substr(idx + 1));
              node_param->other_kernel_params.gridDimZ = gridDimZ;
            }

            // kernel
            std::getline(node_file_stream, line);
            // kernel params
            std::getline(node_file_stream, line);

            {
              std::getline(node_file_stream, line);
              size_t idx = line.find("=");
              int sharedMemBytes = std::stoi(line.substr(idx + 1));
              node_param->other_kernel_params.sharedMemBytes = sharedMemBytes;
            }

            std::getline(node_file_stream, line);
            {
              if (line.find("paramSize =") != std::string::npos) {
                while (1) {
  param:
                  size_t idx = line.find("paramSize") + 11;
                  int param_size = std::stoi(line.substr(idx + 1));
                  node_param->other_kernel_params.paramSizes.push_back(param_size);

                  char *paramPtr = (char*)malloc(param_size);
                  node_param->other_kernel_params.paramPtrs.push_back(paramPtr);
                  int paramByteIdx = 0;
                  while (paramByteIdx < param_size) {
                    std::getline(node_file_stream, line);
                    if (line == "\n") {
                      break;
                    }
                    
                    std::istringstream iss(line);
                    unsigned int byte;
                    while (iss >> std::hex >> byte) {
                      paramPtr[paramByteIdx++] = static_cast<unsigned char>(byte);
                    }
                  }

                  while(1) {
                    std::getline(node_file_stream, line);
                    if (line.find("paramSize") != std::string::npos) {
                      goto param;
                    }
                    if (line.find("=========") != std::string::npos) {
                      goto finish_param;
                    }
                  }
                }
  finish_param:
                if (layer_no == 1) {
                  one_layer_nodes.push_back(node_param);
                }
                return node_param;
              }
            }
          } else {
            TORCH_CHECK(false, "node type not found");
          }
      }
    }
  }
  return nullptr;
}

void CUDAGraph::read_node() {
  int layer = 0;
  std::vector<CUgraphNodeParams*> one_layer_nodes;
  size_t one_layer_nodes_idx = 0;
  if (node_file_stream.is_open()) {
    while(1) {
      CUgraphNodeParams* node_param = read_one_node(layer, one_layer_nodes, one_layer_nodes_idx);
      
      if (node_param == nullptr) {
        break;
      }

      if (node_param->kernel_func_name.find("rotary_embedding_kernel") != std::string::npos) {
        layer++;
      }

      if (node_params.find(node_param->node_idx) != node_params.end()) {
        TORCH_CHECK(false, "node_idx already exists");
      }
      node_params.insert({node_param->node_idx, node_param});
    }

    node_file_stream.close();
  } else {
    SERVERLESS_LOG("open file failed\n");
  }
}

void CUDAGraph::read_dependency() {
  // std::regex dependency_pattern("(\\d+) (\\d+)");

  // if (dependency_file_stream.is_open()) {
  //   std::string line;
  //   while (std::getline(dependency_file_stream, line)) {
  //     {
  //       std::smatch match;
  //       if (std::regex_search(line, match, dependency_pattern)) {
  //           std::string from = match.str(1);
  //           std::string to = match.str(2);
  //           int from_idx = std::stoi(from);
  //           int to_idx = std::stoi(to);

  //           saved_from_idx.push_back(from_idx);
  //           saved_to_idx.push_back(to_idx);
  //       }
  //     }
  //   }

  //   dependency_file_stream.close();

  // } else {
  //   SERVERLESS_LOG("open file failed\n");
  // }
}

void CUDAGraph::replay_mf() {
  uint64_t mf_idx = 0;
  uint64_t malloc_idx = 0;

  bool begin_replay = false;

  uint64_t already_malloc_nums = malloc_addrs->size();

  {
    std::string line;
    std::getline(mf_file_stream, line);
    malloc_idx = std::stoull(line);
  }

  if (mf_file_stream.is_open()) {
    std::string line;
    while (std::getline(mf_file_stream, line)) {
      {        
        if (line.rfind("m", 0) == 0) {
            std::istringstream iss(line.substr(2));
            uint64_t _;
            uint64_t malloc_size;
            iss >> _;
            iss >> malloc_size;

            // use malloc addr to anchor, since free could be reorder
            if (malloc_idx == already_malloc_nums && !begin_replay) {
              begin_replay = true;
              SERVERLESS_LOG("replay from idx: %ld\n", mf_idx);
            }

            SERVERLESS_LOG("mf_idx: %ld, m %ld\n", mf_idx, malloc_size);

            if (begin_replay) {
              auto allocator = c10::cuda::CUDACachingAllocator::get();
              allocator->raw_alloc(malloc_size);
            }
            malloc_idx++;
            mf_idx++;
        }

        if (line.rfind("f", 0) == 0) {
            std::istringstream iss(line.substr(2));
            uint64_t free_idx;
            iss >> free_idx;

            SERVERLESS_LOG("mf_idx: %ld, f %ld\n", mf_idx, free_idx);
            if (begin_replay) {
              auto allocator = c10::cuda::CUDACachingAllocator::get();
              allocator->raw_delete((void*)(*malloc_addrs)[free_idx]);
            }
            mf_idx++;
        }
      }
    }

    mf_file_stream.close();

  } else {
    SERVERLESS_LOG("open file failed\n");
  }

  SERVERLESS_LOG("replay malloc and free success\n");
}

#define LOAD_GRAPH_BEGIN_IDX SAVE_GRAPH_BEGIN_IDX + 1

at::Tensor CUDAGraph::set_output_tensor(const at::Tensor &tensor) {
  /* the addr is changed
     as for the dimension:
      embedding -> hidden_states dimension
      is equal to
      model.forward -> hidden_states dimension
  */
  at::TensorOptions options;
  if (model == Llama2_7B)
    options = at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
  else if (model == Llama2_13B)
    options = at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
  else if (model == Yi_6B)
    options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  else if (model == Yi_9B)
    options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  else if (model == Qwen_14B)
    options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  else if (model == Qwen_7B)
    options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  else if (model == Qwen_4B)
    options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  else if (model == Qwen_1_8B)
    options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  else if (model == Qwen_0_5B)
    options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  else if (model == Falcon_7B)
    options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  else
    TORCH_CHECK(false, "model not found");

  SERVERLESS_LOG("set_output_tensor shape:\n");
  for (auto i : tensor.sizes().vec()) {
    SERVERLESS_LOG("%ld ", i);
  }
  SERVERLESS_LOG("\n");
  return at::from_blob((void*)(result_hidden_states_addr), tensor.sizes().vec(), options);
}

void CUDAGraph::load_kernel_graph(
  const std::vector<CUgraphNode> &saved_nodes) {
  auto load_kernel_graph_start = std::chrono::high_resolution_clock::now();
  
  c10::cuda::CUDACachingAllocator::getAllMallocAddrs(&malloc_addrs);
  {
    read_node();
  }
  {
    read_func_params();
  }
  // {
  //   auto start = std::chrono::high_resolution_clock::now();
  //   read_dependency();
  //   auto end = std::chrono::high_resolution_clock::now();
  //   std::chrono::duration<double> duration = end - start;
  //   std::cout << "dependency Duration: " << duration.count() << " seconds" << std::endl;
  // }
  {
    replay_mf();
  }
  
  // Create an new CUDA graph
  cudaGraph_t newGraph;
  CU_CALL(cuGraphCreate(&newGraph, 0));

  std::vector<CUgraphNode> new_nodes_vec;

  // ctx is the same across all nodes
  // ensured by checkGraphKernelNodeParams
  CUcontext ctx;
  // =========== load already captured nodes (already exclude trigger nodes) ===========
  {
    for (size_t i = 0; i < saved_nodes.size(); i++) {
      CUgraphNode newNode;

      CUgraphNodeType type;
      CU_CALL(cuGraphNodeGetType(saved_nodes[i], &type));

      switch (type) {
        case CU_GRAPH_NODE_TYPE_KERNEL: {
          CUDA_KERNEL_NODE_PARAMS p;
          CU_CALL(cuGraphKernelNodeGetParams(saved_nodes[i], &p));
          CU_CALL(cuGraphAddKernelNode(&newNode, newGraph, nullptr, 0, &p));
          reuseGraphKernelNodeParams(i, p);
          ctx = p.ctx;
          break;
        }
        case CU_GRAPH_NODE_TYPE_MEMSET: {
          CUDA_MEMSET_NODE_PARAMS p;
          CU_CALL(cuGraphMemsetNodeGetParams(saved_nodes[i], &p));
          CU_CALL(cuGraphAddMemsetNode(&newNode, newGraph, nullptr, 0, &p, ctx));
          break;
        }
        default: {
          SERVERLESS_LOG("error node type: %d\n", type);
          TORCH_CHECK(false);
          break;
        }
      }

      new_nodes_vec.push_back(newNode);
    }
  }

  std::set<std::string> mangled_names_fake;

  // ========================= load other nodes =========================
  for (size_t i = LOAD_GRAPH_BEGIN_IDX + saved_nodes.size() - 1; i < SAVE_GRAPH_BEGIN_IDX + node_params.size(); i++) {
    CUgraphNodeParams *param = node_params[i];
    SERVERLESS_LOG("============================\n");
    SERVERLESS_LOG("node idx: %ld, node type: %d\n", i, param->node_type);
    CUgraphNode newNode;

    switch (param->node_type) {
      case CU_GRAPH_NODE_TYPE_KERNEL: {
        CUDA_KERNEL_NODE_PARAMS p;
        loadGraphKernelNodeParams(ctx, p, param->node_idx);
        CU_CALL(cuGraphAddKernelNode(&newNode, newGraph, nullptr, 0, &p));
        // for debug
        // saveGraphKernelNodeParams(p, mangled_names_fake);
        break;
      }
      case CU_GRAPH_NODE_TYPE_MEMSET: {
        CUDA_MEMSET_NODE_PARAMS p;
        loadGraphMemsetNodeParams(p, param->node_idx);
        CU_CALL(cuGraphAddMemsetNode(&newNode, newGraph, nullptr, 0, &p, ctx));
        break;
      }
      default: {
        SERVERLESS_LOG("error node type: %d\n", param->node_type);
        TORCH_CHECK(false);
        break;
      }
    }
    new_nodes_vec.push_back(newNode);
  }

  SERVERLESS_LOG("add %ld new nodes\n", new_nodes_vec.size());

  // =================== get result hidden_states addr ===================
  // last node, first param
  {
    CUgraphNode last_node = new_nodes_vec[new_nodes_vec.size() - 1];
    CUDA_KERNEL_NODE_PARAMS p;
    CU_CALL(cuGraphKernelNodeGetParams(last_node, &p));
    void *params = p.kernelParams;

    if (model == Llama2_7B)
      result_hidden_states_addr = *(uint64_t*)(*(void**)((uint64_t*)params + 0));
    else if (model == Llama2_13B)
      result_hidden_states_addr = *(uint64_t*)(*(void**)((uint64_t*)params + 0));
    else if (model == Yi_6B)
      result_hidden_states_addr = *(uint64_t*)(*(void**)((uint64_t*)params + 0));
    else if (model == Yi_9B)
      result_hidden_states_addr = *(uint64_t*)(*(void**)((uint64_t*)params + 0));
    else if (model == Qwen_14B)
      result_hidden_states_addr = *(uint64_t*)(*(void**)((uint64_t*)params + 0));
    else if (model == Qwen_7B)
      result_hidden_states_addr = *(uint64_t*)(*(void**)((uint64_t*)params + 0));
    else if (model == Qwen_4B)
      result_hidden_states_addr = *(uint64_t*)(*(void**)((uint64_t*)params + 0));
    else if (model == Qwen_1_8B)
      result_hidden_states_addr = *(uint64_t*)(*(void**)((uint64_t*)params + 0));
    else if (model == Qwen_0_5B)
      result_hidden_states_addr = *(uint64_t*)(*(void**)((uint64_t*)params + 0));
    else if (model == Falcon_7B)
      result_hidden_states_addr = *(uint64_t*)(*(void**)((uint64_t*)params + 7));
    else
      TORCH_CHECK(false, "model not found");
  }

  // ========================= load dependencies =========================
  CUgraphNode saved_from[MAX_NODE_AND_EDGE_NUM];
  CUgraphNode saved_to[MAX_NODE_AND_EDGE_NUM];

  for (size_t i = 0; i < new_nodes_vec.size() - 1; i++) {
    // saved_from[i] = new_nodes_vec[saved_from_idx[i] - SAVE_GRAPH_BEGIN_IDX];
    // saved_to[i] = new_nodes_vec[saved_to_idx[i] - SAVE_GRAPH_BEGIN_IDX];
    // linear dependency, already checked when save
    saved_from[i] = new_nodes_vec[i];
    saved_to[i] = new_nodes_vec[i + 1];
  }

  SERVERLESS_LOG("load %ld dependencies\n", new_nodes_vec.size() - 1);

  CU_CALL(cuGraphAddDependencies(newGraph, saved_from, saved_to, new_nodes_vec.size() - 1));

  // ========================= replace cuda graph =========================

  graph_ = newGraph;

  SERVERLESS_LOG("load cuda graph success\n");
  auto load_kernel_graph_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = load_kernel_graph_end - load_kernel_graph_start;
  std::cout << "load_cuda_graph Duration: " << duration.count() << " seconds" << std::endl;
}

// =================================== load cuda graph API finish ===================================


void CUDAGraph::save_cuda_graph() {
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11030)|| (defined(USE_ROCM) && ROCM_VERSION >= 50600)
  if (_cuda_graphs_save) {
    TORCH_WARN("DEBUG: calling save_cuda_graph()");
    constructFunc2Lib();

    _save_cuda_graph();

    // TORCH_CHECK(false, "Stop here");
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with torch._C._cuda_enable_graphs_save_mode");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

void CUDAGraph::reset() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this CUDAGraph, the generator,
  // and the allocator could end up in all kinds of weird states depending where failure occurred.
  // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
  // a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CUDAGraph::pool() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
  return mempool_id_;
}

CUDAGraph::~CUDAGraph() {
  reset();
}

} // namespace at::cuda

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>

#include <mutex>
#include <map>
#include <string>
#include <fstream>
#include <set>

namespace at {

struct CUDAGeneratorImpl;

namespace cuda {

#define CU_CALL(x) do { CUresult result = x; if (result != CUDA_SUCCESS) { printf("%s:%s:%d CUDA error: %d\n", __FILE__, __func__, __LINE__, result); exit(-1); } } while(0)

// Standalone way to get a unique mempool id usable as a pool=... argument
// to CUDAGraph::capture_begin
TORCH_CUDA_CPP_API MempoolId_t graph_pool_handle();

struct CUgraphNodeParams_t {
  int node_idx;
  CUgraphNodeType type;

  CUDA_KERNEL_NODE_PARAMS kernel_params;
  CUDA_MEMSET_NODE_PARAMS memset_params;
};

struct memset_node_params {
  int dst_malloc_idx;
  uint64_t elementSize;
  uint64_t height;
  uint64_t pitch;
  uint64_t value;
  uint64_t width;
};

struct other_kernel_node_params {
  unsigned int  blockDimX;
  unsigned int  blockDimY;
  unsigned int  blockDimZ;
  unsigned int  gridDimX;
  unsigned int  gridDimY;
  unsigned int  gridDimZ;
  unsigned int  sharedMemBytes;

  std::vector<int> malloc_idxs;
  std::vector<int> paramSizes;
  std::vector<char*> paramPtrs;
};

struct CUgraphNodeParams {
  int node_type;
  int node_idx;
  int corresponding_layer_one_idx;

  std::string kernel_func_name;

  memset_node_params memset_params;
  other_kernel_node_params other_kernel_params;
};

enum model_type {
  Llama2_7B,
  Llama2_13B,
  Yi_6B,
  Yi_9B,
  // Bloom not support
  Qwen_14B,
  Qwen_7B,
  Qwen_4B,
  Qwen_1_8B,
  Qwen_0_5B,
  // Falcon_1B not support
  Falcon_7B,
};

struct TORCH_CUDA_CPP_API CUDAGraph {
  CUDAGraph();
  ~CUDAGraph();

  void set_info_files(const std::string &dependency_file, const std::string &func_params_file, const std::string &node_file, const std::string &mf_file);

  static void inc_pending_event_queries();
  static void dec_pending_event_queries();
  static int num_pending_event_queries();
  void capture_begin(MempoolId_t pool={0, 0}, cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal, bool load_graph = false);
  void capture_end();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);

  void enable_save_mode();
  void set_model(const std::string& model);
  void save_cuda_graph();
  void _save_cuda_graph();
  void constructFunc2Lib();
  void loadCublassModule(const std::vector<cudaGraphNode_t> &node);
  void reuseGraphKernelNodeParams(int node_idx, const CUDA_KERNEL_NODE_PARAMS &p);
  void saveGraphKernelNodeParams(const CUDA_KERNEL_NODE_PARAMS &pNodeParams);
  void loadGraphKernelNodeFunc_fromCuBlass(const char *func_name, CUfunction &f);
  void loadGraphKernelNodeFunc_fromSO(const char *func_name, CUfunction &f, const std::string &libraryName);
  void loadGraphKernelNodeFunc(const char *func_name, CUfunction &f);
  void loadGraphMemsetNodeParams(CUDA_MEMSET_NODE_PARAMS &p, int node_idx);
  void loadGraphKernelNodeParams(const CUcontext &ctx, CUDA_KERNEL_NODE_PARAMS &p, int node_idx);
  void load_kernel_graph (
    const std::vector<CUgraphNode> &saved_nodes);
  void printTensorAddr(const at::Tensor &tensor);
  void loadGraphKernelNodeFuncParams(const char *func_name, void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_fromCuBlass(const char *func_name, void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_ampere(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_ampere_208(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_sm80(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_cublasSplitKreduce_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_fromOtherKernel(const char *func_name, void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_static_load(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_paged_attention_v1_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_paged_attention_v2_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_paged_attention_v2_reduce_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_fused_add_rms_norm_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_reshape_and_cache_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_rotary_embedding_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_silu_and_mul_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_rms_norm_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_layer_norm_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_vectorized_elementwise_kernel_TensorIteratorBase(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_vectorized_elementwise_kernel(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_cutlass_80_tensorop(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_cutlass_80_tensorop_360(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_cublasGemv(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_cublasGemv_152(void** &params, int node_idx);
  void loadGraphKernelNodeFuncParams_cublasGemv_152_2(void** &params, int node_idx);

  uint64_t get_key_addr(uint64_t addr);
  uint64_t get_value_addr(uint64_t addr);

  CUgraphNodeParams* read_one_node(int layer_no, std::vector<CUgraphNodeParams*> &one_layer_nodes, size_t &one_layer_nodes_idx);
  void read_node();
  void read_func_params();
  void read_dependency();

  void replay_mf();

  at::Tensor set_output_tensor(const at::Tensor &tensor);

  protected:
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
#endif
  bool load_graph_;
  std::vector<CUmodule> cublass_mods;
  std::map<std::string, std::string> func2lib_;
  std::map<int, CUgraphNodeParams*> node_params;
  std::map<int, void **> kernel_node_params;
  std::vector<uint64_t> *malloc_addrs;

  std::vector<int> saved_from_idx;
  std::vector<int> saved_to_idx;

  uint64_t result_hidden_states_addr;

  static std::atomic<int> pending_event_queries;

  std::vector<void*> cuda_addr_for_sm80_addrs;
  size_t cuda_addr_for_sm80_idx{0};

  std::vector<void*> cublasSplitKreduce_kernel_addrs;
  size_t cublasSplitKreduce_kernel_idx{0};

  std::map<std::string, CUfunction> func2ptr_;

  std::ifstream dependency_file_stream;
  std::ifstream func_params_file_stream;
  std::ifstream node_file_stream;
  std::ifstream mf_file_stream;

  enum model_type model;
  int key_offset;
  int value_offset;

  // internal states so reset() can do its best cleaning up
  // Set to true in capture_end if cudaStreamEndCapture succeeded
  // Set back to false soon after, when graph_ is consumed by cudaGraphInstantiate
  // to create graph_exec_, then graph_ is deleted
  bool has_graph_ = false;
  // Set to true in capture_end if cudaGraphInstantiate succeeded
  bool has_graph_exec_ = false;

  // uuid of this instance's current capture, retrieved from Cuda
  CaptureId_t id_;

  // uuid used to request a particular private mempool from CUDACachingAllocator.
  // By default, this will be set to {id_, 0}.
  //
  // If capture_begin is called with "pool=other_graph.pool()", this graph's mempool_id_
  // will be set to the other graph's mempool_id_, and therefore share a mempool with the
  // other graph.
  //
  // If capture_begin is called with "pool=handle" where "handle" came from graph_pool_handle(),
  // it will share a mempool with any other captures that used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  MempoolId_t mempool_id_;

  // Stream on which capture began
  at::cuda::CUDAStream capture_stream_;

  // Default generator on device where capture began
  at::CUDAGeneratorImpl* capture_gen_;

  // Device where capture occurred. Right now, for simplicity, we require all ops
  // in a capture to run on the same device, but this is a limitation of CUDAGraph,
  // not CUDA itself.  We can straightforwardly modify CUDAGraph to support multi-device
  // captures if needed.
  int capture_dev_;

  // RNG state trackers
  at::Tensor seed_extragraph_;
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;
};

} // namespace cuda
} // namespace at

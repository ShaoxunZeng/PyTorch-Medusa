#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/cuda/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer
// (csrc/Module.cpp) I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m.def("_graph_pool_handle", &::at::cuda::graph_pool_handle);

  shared_ptr_class_<::at::cuda::CUDAGraph>(torch_C_m, "_CUDAGraph")
      .def(py::init<>())
      .def(
          "capture_begin",
          [](::at::cuda::CUDAGraph& self,
             c10::optional<c10::cuda::MempoolId_t> pool_opt,
             std::string capture_error_mode,
             bool load_graph = false) {
            cudaStreamCaptureMode capture_mode;
            c10::cuda::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::cuda::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = cudaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = cudaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = cudaStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode, load_graph);
          },
          py::arg("pool"),
          py::arg("capture_error_mode"),
          py::arg("load_graph"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::capture_end))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::replay))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::pool))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::debug_dump))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::enable_debug_mode))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::debug_dump),
          py::arg("debug_path"))
      .def(
          "enable_save_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::enable_save_mode))
      .def(
          "printTensorAddr",
            torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::printTensorAddr),
          py::arg("tensor"))
      .def(
          "set_output_tensor",
            torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::set_output_tensor),
          py::arg("tensor"))
      .def(
            "set_info_files",
            torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::set_info_files),
          py::arg("dependency_file"),
          py::arg("func_param_file"),
          py::arg("node_file"),
          py::arg("mf_file"))
      .def(
            "set_model",
            torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::set_model),
          py::arg("model"));
}

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[LiteMono/TRT] %s\n", msg);
    }
  }
};

template <typename T>
struct TrtDestroy {
  void operator()(T* ptr) const noexcept {
    delete ptr;
  }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDestroy<T>>;

void set_error(char* dst, std::size_t cap, const std::string& message) {
  if (!dst || cap == 0) return;
  std::snprintf(dst, cap, "%s", message.c_str());
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(what) + ": " + cudaGetErrorString(status));
  }
}

std::size_t dtype_size(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL: return 1;
    default: throw std::runtime_error("unsupported TensorRT binding dtype");
  }
}

std::size_t volume(const nvinfer1::Dims& dims) {
  std::size_t value = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      throw std::runtime_error("dynamic/invalid binding shape is not supported");
    }
    value *= static_cast<std::size_t>(dims.d[i]);
  }
  return value;
}

class LiteMonoEngine {
 public:
  explicit LiteMonoEngine(const char* engine_path) {
    if (!engine_path || !*engine_path) {
      throw std::runtime_error("empty TensorRT engine path");
    }

    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open TensorRT engine");
    const std::streamsize size = file.tellg();
    if (size <= 0) throw std::runtime_error("TensorRT engine is empty");
    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (!file.read(bytes.data(), size)) {
      throw std::runtime_error("failed to read TensorRT engine");
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("createExecutionContext failed");

    input_index_ = engine_->getBindingIndex("input");
    disp_index_ = engine_->getBindingIndex("disp_output");
    if (input_index_ < 0 || disp_index_ < 0) {
      throw std::runtime_error(
          "engine must expose bindings named input and disp_output");
    }
    if (!engine_->bindingIsInput(input_index_) ||
        engine_->bindingIsInput(disp_index_)) {
      throw std::runtime_error("unexpected TensorRT input/output bindings");
    }
    if (engine_->getBindingDataType(input_index_) !=
            nvinfer1::DataType::kFLOAT ||
        engine_->getBindingDataType(disp_index_) !=
            nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("Lite-Mono input/disp_output must be FP32 bindings");
    }

    const auto input_dims = engine_->getBindingDimensions(input_index_);
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 ||
        input_dims.d[1] != 3) {
      throw std::runtime_error("expected Lite-Mono input shape [1,3,H,W]");
    }
    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];

    disp_elements_ = volume(engine_->getBindingDimensions(disp_index_));
    if (disp_elements_ !=
        static_cast<std::size_t>(input_h_) *
            static_cast<std::size_t>(input_w_)) {
      throw std::runtime_error("unexpected disp_output element count");
    }

    const int binding_count = engine_->getNbBindings();
    buffers_.assign(binding_count, nullptr);
    bytes_.assign(binding_count, 0);
    try {
      for (int i = 0; i < binding_count; ++i) {
        bytes_[i] = volume(engine_->getBindingDimensions(i)) *
                    dtype_size(engine_->getBindingDataType(i));
        check_cuda(cudaMalloc(&buffers_[i], bytes_[i]), "cudaMalloc");
      }
      check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    } catch (...) {
      release_cuda();
      throw;
    }
  }

  ~LiteMonoEngine() { release_cuda(); }

  LiteMonoEngine(const LiteMonoEngine&) = delete;
  LiteMonoEngine& operator=(const LiteMonoEngine&) = delete;

  int input_h() const { return input_h_; }
  int input_w() const { return input_w_; }
  std::size_t disp_elements() const { return disp_elements_; }

  void infer(const float* input, float* disp) {
    if (!input || !disp) throw std::runtime_error("null inference buffer");
    check_cuda(
        cudaMemcpyAsync(buffers_[input_index_], input, bytes_[input_index_],
                        cudaMemcpyHostToDevice, stream_),
        "copy input H2D");

    if (!context_->enqueueV2(buffers_.data(), stream_, nullptr)) {
      throw std::runtime_error("TensorRT enqueueV2 failed");
    }

    check_cuda(
        cudaMemcpyAsync(disp, buffers_[disp_index_], bytes_[disp_index_],
                        cudaMemcpyDeviceToHost, stream_),
        "copy disp_output D2H");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  }

 private:
  void release_cuda() noexcept {
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    for (void*& ptr : buffers_) {
      if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
      }
    }
  }

  Logger logger_;
  TrtPtr<nvinfer1::IRuntime> runtime_;
  TrtPtr<nvinfer1::ICudaEngine> engine_;
  TrtPtr<nvinfer1::IExecutionContext> context_;
  std::vector<void*> buffers_;
  std::vector<std::size_t> bytes_;
  cudaStream_t stream_ = nullptr;
  int input_index_ = -1;
  int disp_index_ = -1;
  int input_h_ = 0;
  int input_w_ = 0;
  std::size_t disp_elements_ = 0;
};

}  // namespace

extern "C" {

void* litemono_create(const char* engine_path, char* error, std::size_t error_cap) {
  try {
    return new LiteMonoEngine(engine_path);
  } catch (const std::exception& exc) {
    set_error(error, error_cap, exc.what());
    return nullptr;
  }
}

int litemono_get_input_hw(void* handle, int* height, int* width,
                          char* error, std::size_t error_cap) {
  try {
    if (!handle || !height || !width) throw std::runtime_error("invalid handle/output");
    auto* engine = static_cast<LiteMonoEngine*>(handle);
    *height = engine->input_h();
    *width = engine->input_w();
    return 0;
  } catch (const std::exception& exc) {
    set_error(error, error_cap, exc.what());
    return -1;
  }
}

int litemono_get_output_elements(void* handle, std::size_t* elements,
                                 char* error, std::size_t error_cap) {
  try {
    if (!handle || !elements) throw std::runtime_error("invalid handle/output");
    *elements = static_cast<LiteMonoEngine*>(handle)->disp_elements();
    return 0;
  } catch (const std::exception& exc) {
    set_error(error, error_cap, exc.what());
    return -1;
  }
}

int litemono_infer(void* handle, const float* input_chw, float* disp,
                   char* error, std::size_t error_cap) {
  try {
    if (!handle) throw std::runtime_error("invalid Lite-Mono handle");
    static_cast<LiteMonoEngine*>(handle)->infer(input_chw, disp);
    return 0;
  } catch (const std::exception& exc) {
    set_error(error, error_cap, exc.what());
    return -1;
  }
}

void litemono_destroy(void* handle) {
  delete static_cast<LiteMonoEngine*>(handle);
}

}  // extern "C"

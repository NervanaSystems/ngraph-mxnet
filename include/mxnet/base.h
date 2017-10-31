/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file base.h
 * \brief configuation of mxnet as well as basic data structure.
 */
#ifndef MXNET_BASE_H_
#define MXNET_BASE_H_

#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <dmlc/parameter.h>
#include <mshadow/tensor.h>
// nnvm headers for symbolic construction.
#include <nnvm/op.h>
#include <nnvm/tuple.h>
#include <nnvm/symbolic.h>
#include <string>

/*!
 *\brief whether to use opencv support
 */
#ifndef MXNET_USE_OPENCV
#define MXNET_USE_OPENCV 1
#endif

/*!
 *\brief whether to use cuda support
 */
#ifndef MXNET_USE_CUDA
#define MXNET_USE_CUDA MSHADOW_USE_CUDA
#endif

/*!
 *\brief whether to use cudnn library for convolution
 */
#ifndef MXNET_USE_CUDNN
#define MXNET_USE_CUDNN MSHADOW_USE_CUDNN
#endif

/*!
 *\brief whether to use cusolver library
 */
#ifndef MXNET_USE_CUSOLVER
#define MXNET_USE_CUSOLVER MSHADOW_USE_CUSOLVER
#endif

/*! \brief Error message for using gpu when MXNET_USE_CUDA==0 */
#define MXNET_GPU_NOT_ENABLED_ERROR  "GPU is not enabled"

/*!
 * \brief define compatible keywords in g++
 *  Used to support g++-4.6 and g++4.7
 */
#if DMLC_USE_CXX11 && defined(__GNUC__) && !defined(__clang_version__)
#if __GNUC__ == 4 && __GNUC_MINOR__ < 8
#error "Currently we need g++ 4.8 or higher to fully support c++11 features"
#define override
#define final
#endif
#endif

/*!
 * \brief define dllexport for Visual Studio
 */
#ifdef _MSC_VER
#ifdef MXNET_EXPORTS
#define MXNET_API __declspec(dllexport)
#else
#define MXNET_API __declspec(dllimport)
#endif
#else
#define MXNET_API
#endif

/*!
 * \brief define prediction only
 */
#ifndef MXNET_PREDICT_ONLY
#define MXNET_PREDICT_ONLY 0
#endif

/*!
 * \brief define operator message for profiler
 */
#if MXNET_USE_PROFILER
#define PROFILER_MESSAGE(msg)     msg
#else
#define PROFILER_MESSAGE(msg)     nullptr
#endif

/*! \brief major version */
#define MXNET_MAJOR 0
/*! \brief minor version */
#define MXNET_MINOR 12
/*! \brief patch version */
#define MXNET_PATCH 0
/*! \brief mxnet version */
#define MXNET_VERSION (MXNET_MAJOR*10000 + MXNET_MINOR*100 + MXNET_PATCH)
/*! \brief helper for making version number */
#define MXNET_MAKE_VERSION(major, minor, patch) ((major)*10000 + (minor)*100 + patch)
/*!
 * \brief define function name as profiler message
 */
#define PROFILER_MESSAGE_FUNCNAME PROFILER_MESSAGE(__FUNCTION__)

/*! \brief namespace of mxnet */
namespace mxnet {
/*! \brief mxnet cpu */
typedef mshadow::cpu cpu;
/*! \brief mxnet gpu */
typedef mshadow::gpu gpu;
/*! \brief index type usually use unsigned */
typedef mshadow::index_t index_t;
/*! \brief data type that will be used to store ndarray */
typedef mshadow::default_real_t real_t;
/*! \brief Shape data structure used to record shape information */
using TShape = nnvm::TShape;
/*! \brief operator structure from NNVM */
using Op = nnvm::Op;

/*! \brief Context information about the execution environment */
struct Context {
  /*! \brief Type of device */
  enum DeviceType {
    kCPU = cpu::kDevMask,
    kGPU = gpu::kDevMask,
    kCPUPinned = 3
  };
  /*! \brief the device type we run the op on */
  DeviceType dev_type;
  /*! \brief device id we are going to run it on */
  int32_t dev_id;
  /*! \brief default constructor */
  Context() : dev_type(kCPU), dev_id(0) {}
  /*!
   * \brief Get corresponding device mask
   * \return cpu::kDevMask or gpu::kDevMask
   */
  inline int dev_mask() const {
    if (dev_type == kCPUPinned) return cpu::kDevMask;
    return dev_type;
  }
  /*!
   * \brief Comparator, used to enable Context as std::map key.
   * \param b another context to compare
   * \return compared result
   */
  inline bool operator<(const Context &b) const;
  /*!
   * \brief check if current context equals another one
   * \param b another context to compare
   * \return whether dev mask and id are same
   */
  inline bool operator==(const Context &b) const {
    return dev_type == b.dev_type && dev_id == b.dev_id;
  }
  /*!
   * \brief check if current context not equals another one
   * \param b another context to compare
   * \return whether they are not the same
   */
  inline bool operator!=(const Context &b) const {
    return !(*this == b);
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  inline void Save(dmlc::Stream *strm) const {
    strm->Write(&dev_type, sizeof(dev_type));
    strm->Write(&dev_id, sizeof(dev_id));
  }
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  inline bool Load(dmlc::Stream *strm) {
    if (strm->Read(&dev_type, sizeof(dev_type)) != sizeof(dev_type)) return false;
    if (strm->Read(&dev_id, sizeof(int32_t)) != sizeof(int32_t)) return false;
    return true;
  }
  /*! \brief the maximal device type */
  static const int32_t kMaxDevType = 4;
  /*! \brief the maximal device index */
  static const int32_t kMaxDevID = 16;
  /*!
   * \brief Create a new context.
   * \param dev_type device type.
   * \param dev_id device id. -1 for current device.
   */
  inline static Context Create(DeviceType dev_type, int32_t dev_id = -1);
  /*! \return CPU Context */
  inline static Context CPU(int32_t dev_id = 0);
  /*!
   * Create a GPU context.
   * \param dev_id the device id.
   * \return GPU Context. -1 for current GPU.
   */
  inline static Context GPU(int32_t dev_id = -1);
  /*!
   * Create a pinned CPU context.
   * \param dev_id the device id for corresponding GPU.
   * \return Pinned CPU context. -1 for current GPU.
   */
  inline static Context CPUPinned(int32_t dev_id = -1);
  /*!
   * Create a context from string of the format [cpu|gpu|cpu_pinned](n)
   * \param str the string pattern
   * \return Context
   */
  inline static Context FromString(std::string str);
};

/*!
 * \brief execution time context.
 *  The information needed in runtime for actual execution.
 */
struct RunContext {
  /*! \brief base Context */
  Context ctx;
  /*!
   * \brief the stream of the device, can be NULL or Stream<gpu>* in GPU mode
   */
  void *stream;
  /*!
   * \brief get mshadow stream from Context
   * \return the mshadow stream
   * \tparam xpu the device type of the stream
   */
  template<typename xpu>
  inline mshadow::Stream<xpu>* get_stream() const {
    return static_cast<mshadow::Stream<xpu>*>(stream);
  }
  /*! \brief get the base Context from RunContext */
  inline const Context& get_ctx() const {
    return ctx;
  }
};
}  // namespace mxnet

//! \cond Doxygen_Suppress
namespace mxnet {
// implementing Context
inline bool Context::operator<(const Context &b) const {
  if (dev_type == b.dev_type) {
    return dev_id < b.dev_id;
  } else {
    return dev_type < b.dev_type;
  }
}
inline Context Context::Create(DeviceType dev_type, int32_t dev_id) {
  Context ctx;
  ctx.dev_type = dev_type;
  if (dev_id < 0) {
    ctx.dev_id = 0;
    if (dev_type != kCPU) {
#if MXNET_USE_CUDA
      CHECK_EQ(cudaGetDevice(&ctx.dev_id), cudaSuccess);
#else
      LOG(FATAL) << "Please compile with CUDA enabled for cuda features";
#endif
    }
  } else {
    ctx.dev_id = dev_id;
  }
  return ctx;
}
inline Context Context::CPU(int32_t dev_id) {
  return Create(kCPU, dev_id);
}

inline Context Context::CPUPinned(int32_t dev_id) {
  return Create(kCPUPinned, dev_id);
}

inline Context Context::GPU(int32_t dev_id) {
  return Create(kGPU, dev_id);
}

inline Context Context::FromString(std::string str) {
  Context ret;
  try {
    std::string::size_type l = str.find('(');
    CHECK_NE(l, std::string::npos);
    std::string::size_type r = str.find(')');
    CHECK_EQ(r, str.length()-1);

    std::string type = str.substr(0, l);
    int id = std::stoi(str.substr(l+1, r-l-1));
    if (type == "cpu") {
      ret = CPU(id);
    } else if (type == "gpu") {
      ret = GPU(id);
    } else if (type == "cpu_pinned") {
      ret = CPUPinned(id);
    } else {
      LOG(FATAL) << "Invalid context string " << str;
    }
  } catch (...) {
    LOG(FATAL) << "Invalid context string " << str;
  }
  return ret;
}

inline std::ostream& operator<<(std::ostream &out, const Context &ctx) {
  if (ctx.dev_type == Context::kCPU) {
    out << "cpu(";
  } else if (ctx.dev_type == Context::kGPU) {
    out << "gpu(";
  } else if (ctx.dev_type == Context::kCPUPinned) {
    out << "cpu_pinned(";
  } else {
    out << "unknown(";
  }
  out << ctx.dev_id << ")";
  return out;
}

// describe op registration point
#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
#define MXNET_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" STRINGIZE(__LINE__))
#define ADD_FILELINE "\n\nDefined in " __FILE__ ":L" STRINGIZE(__LINE__)

}  // namespace mxnet

#include "./tensor_blob.h"
//! \endcond
#endif  // MXNET_BASE_H_

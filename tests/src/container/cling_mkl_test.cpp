#!/usr/bin/env cling -std=c++17
#pragma cling add_include_path("/home/jonas/phd/projects/rl_for_control/backprop_tools/include")
#pragma cling add_include_path("/opt/intel/oneapi/mkl/latest/include")
#pragma cling add_library_path("/opt/intel/oneapi/mkl/latest/lib/intel64/")
#pragma cling add_library_path("/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin")

#define BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mux.h>

#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#pragma cling load("mkl_core")
#pragma cling load("mkl_intel_thread")
#pragma cling load("mkl_intel_ilp64")
#pragma cling load("iomp5")
#pragma cling load("libpthread.so.0")
#pragma cling load("libm.so.6")
#pragma cling load("libdl.so.2")
#endif

namespace bpt = backprop_tools;

using DEVICE = bpt::devices::DefaultCPU;
using DEVICE_MKL = bpt::devices::DefaultCPU_MKL;
using T = float;
using TI = typename DEVICE::index_t;

bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 2, 2>> A, B, C, C_mkl;
DEVICE device;
DEVICE_MKL device_mkl;
auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 1);

bpt::malloc(device, A);
bpt::malloc(device, B);
bpt::malloc(device, C);
bpt::malloc(device, C_mkl);
bpt::randn(device, A, rng);
bpt::randn(device, B, rng);

bpt::print_python_literal(device, A);
bpt::print_python_literal(device, B);

bpt::multiply(device, A, B, C);
bpt::print_python_literal(device, C);

bpt::multiply(device_mkl, A, B, C_mkl);
bpt::print_python_literal(device, C_mkl);

auto diff = bpt::abs_diff(device, C, C_mkl);
std::cout << "diff: " << diff << std::endl;
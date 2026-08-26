file(GLOB ENGINE_SOURCES "*.cpp")
add_subdirectory(common)
add_subdirectory(transport)

# EFA library path is set globally via common.cmake (LIBFABRIC_LIB_DIR)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ---------------------------------------------------------------------------
# AMDGPU-MPU ABI 1.8
# ---------------------------------------------------------------------------
set(AMDGPU_MPU_ROOT "" CACHE PATH
    "AMDGPU-MPU installation/prefix containing include/ and lib/")
set(AMDGPU_MPU_INCLUDE_DIR "" CACHE PATH
    "Directory containing the ABI 1.8 libamdgpu_mpu.h")
set(AMDGPU_MPU_LIBRARY "" CACHE FILEPATH
    "ABI 1.8 libamdgpu_mpu.so")

if(AMDGPU_MPU_ROOT)
  if(NOT AMDGPU_MPU_INCLUDE_DIR)
    set(AMDGPU_MPU_INCLUDE_DIR "${AMDGPU_MPU_ROOT}/include")
  endif()
  if(NOT AMDGPU_MPU_LIBRARY)
    set(AMDGPU_MPU_LIBRARY "${AMDGPU_MPU_ROOT}/lib/libamdgpu_mpu.so")
  endif()
endif()

if(NOT AMDGPU_MPU_INCLUDE_DIR)
  find_path(AMDGPU_MPU_INCLUDE_DIR
    NAMES libamdgpu_mpu.h
    PATHS /usr/local/include /usr/include
    NO_DEFAULT_PATH)
endif()

if(NOT AMDGPU_MPU_LIBRARY)
  find_library(AMDGPU_MPU_LIBRARY
    NAMES amdgpu_mpu libamdgpu_mpu
    PATHS /usr/local/lib /usr/lib /usr/lib64
    NO_DEFAULT_PATH)
endif()

if(NOT AMDGPU_MPU_INCLUDE_DIR OR
   NOT EXISTS "${AMDGPU_MPU_INCLUDE_DIR}/libamdgpu_mpu.h")
  message(FATAL_ERROR
    "AMDGPU-MPU ABI 1.8 header not found. "
    "Set AMDGPU_MPU_ROOT or AMDGPU_MPU_INCLUDE_DIR.")
endif()

if(NOT AMDGPU_MPU_LIBRARY OR NOT EXISTS "${AMDGPU_MPU_LIBRARY}")
  message(FATAL_ERROR
    "AMDGPU-MPU ABI 1.8 library not found. "
    "Set AMDGPU_MPU_ROOT or AMDGPU_MPU_LIBRARY.")
endif()

if(USE_HIP)
  hipify_files(ENGINE_SOURCES)
endif()

add_library(transfer_engine ${ENGINE_SOURCES} $<TARGET_OBJECTS:transport>)

# The MPU adapter consumes the public ABI header and shared library directly.
# There is no dlopen/dlsym layer: ABI 1.8 is a real compile/link dependency.
target_include_directories(transfer_engine PRIVATE "${AMDGPU_MPU_INCLUDE_DIR}")
target_link_libraries(transfer_engine PRIVATE "${AMDGPU_MPU_LIBRARY}")
get_filename_component(AMDGPU_MPU_LIBRARY_DIR
  "${AMDGPU_MPU_LIBRARY}" DIRECTORY)
set_property(TARGET transfer_engine APPEND PROPERTY
  BUILD_RPATH "${AMDGPU_MPU_LIBRARY_DIR}")
set_property(TARGET transfer_engine APPEND PROPERTY
  INSTALL_RPATH "${AMDGPU_MPU_LIBRARY_DIR}")

if(BUILD_SHARED_LIBS)
  install(TARGETS transfer_engine DESTINATION lib)
endif()

add_compile_definitions(transfer_engine PUBLIC MOONCAKE_USE_ETCD CMAKE_INCLUDE)
if(USE_ETCD)
  if(USE_ETCD_LEGACY)
    if(USE_STATIC_ETCD_CPP_API)
      find_package(PkgConfig REQUIRED)
      pkg_check_modules(GRPCPP REQUIRED grpc++)
      pkg_check_modules(GRPC REQUIRED grpc)
      target_link_libraries(
        transfer_engine PUBLIC etcd-cpp-api-core protobuf ${GRPC_LDFLAGS}
                               ${GRPCPP_LDFLAGS})
    else()
      target_link_libraries(transfer_engine PUBLIC etcd-cpp-api)
    endif()
  else()
    add_dependencies(transfer_engine build_etcd_wrapper)
    set(ETCD_WRAPPER_LIB
        ${CMAKE_CURRENT_BINARY_DIR}/../../mooncake-common/etcd/libetcd_wrapper.so
    )
    target_link_libraries(transfer_engine PRIVATE ${ETCD_WRAPPER_LIB})
  endif()
endif()
if(USE_REDIS)
  target_link_libraries(transfer_engine PUBLIC hiredis)
endif()
if(USE_HTTP)
  find_package(CURL REQUIRED)
  target_link_libraries(transfer_engine PUBLIC ${CURL_LIBRARIES})
endif()
target_link_libraries(
  transfer_engine
  PUBLIC base
         transport
         rdma_transport
         mooncake_common
         ibverbs
         glog::glog
         gflags::gflags
         pthread
         dl
         JsonCpp::JsonCpp
         numa
         asio_shared
         yalantinglibs::yalantinglibs)

if(USE_BAREX)
  target_link_libraries(transfer_engine PUBLIC barex_transport)
endif()

if(USE_CUDA)
  target_include_directories(transfer_engine PRIVATE /usr/local/cuda/include)
  if(USE_CXI)
    target_link_libraries(transfer_engine PUBLIC cuda cudart rt)
  else()
    target_link_libraries(transfer_engine PUBLIC cuda cudart rt mlx5)
  endif()
  if(USE_NVMEOF)
    target_link_libraries(transfer_engine PUBLIC nvmeof_transport cufile)
  endif()
endif()

if(USE_MACA)
  if(NOT DEFINED MACA_RUNTIME_LIBS)
    set(MACA_RUNTIME_LIBS mcruntime mxc-runtime64 rt)
  endif()
  target_include_directories(transfer_engine PRIVATE ${MACA_INCLUDE_DIR})
  target_link_libraries(transfer_engine PUBLIC ${MACA_RUNTIME_LIBS})
endif()

if(USE_MUSA)
  target_include_directories(transfer_engine PRIVATE /usr/local/musa/include)
  target_link_libraries(transfer_engine PUBLIC musa musart rt)
endif()

if(USE_HYGON)
  target_link_libraries(transfer_engine PUBLIC cuda cudart rt)
endif()

if(USE_COREX)
  target_include_directories(transfer_engine PRIVATE ${COREX_INCLUDE_DIR})
  target_link_libraries(transfer_engine PUBLIC cuda cudart rt)
endif()

if(USE_HIP)
  target_include_directories(transfer_engine PRIVATE ${HIP_INCLUDE_DIRS})
  target_link_libraries(transfer_engine PUBLIC hip::host rt)

  option(USE_HIP_DMABUF "Enable HIP dmabuf MR registration" ON)
  if(USE_HIP_DMABUF)
    find_package(hsa-runtime64 CONFIG)
    if(hsa-runtime64_FOUND)
      target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)
      target_link_libraries(rdma_transport PUBLIC hsa-runtime64::hsa-runtime64)
      target_link_libraries(transfer_engine PUBLIC hsa-runtime64::hsa-runtime64)
      message(STATUS "HIP dmabuf MR registration enabled (hsa-runtime64 found)")
    else()
      message(STATUS "HIP dmabuf MR registration disabled (hsa-runtime64 not found)")
    endif()
  else()
    message(STATUS "HIP dmabuf MR registration disabled (USE_HIP_DMABUF=OFF)")
  endif()
endif()

if(USE_MLU)
  target_link_libraries(transfer_engine PUBLIC cnrt cndrv)
endif()

if(USE_ASCEND)
  target_link_libraries(transfer_engine PUBLIC ascendcl hccl ascend_transport MPI::MPI)
endif()

if(USE_ASCEND_DIRECT)
  target_link_libraries(transfer_engine PUBLIC ascend_transport)
endif()

if(USE_UBSHMEM)
  target_link_libraries(transfer_engine PUBLIC ascend_transport)
endif()

if(USE_ASCEND_HETEROGENEOUS)
  file(GLOB ASCEND_TOOLKIT_ROOT "/usr/local/Ascend/ascend-toolkit/latest/*-linux")
  set(ASCEND_LIB_DIR "${ASCEND_TOOLKIT_ROOT}/lib64")
  link_directories(${ASCEND_LIB_DIR})
  target_link_libraries(transfer_engine PUBLIC ascendcl ascend_transport)
endif()

if(USE_TENT)
  add_compile_definitions(transfer_engine PUBLIC USE_TENT)
  target_link_libraries(transfer_engine PUBLIC tent_link_group)
endif()

if(USE_INTRA_NVLINK)
  message(STATUS "Enabled USE_INTRA_NVLINK support")
  target_compile_definitions(transfer_engine PUBLIC USE_INTRA_NVLINK)
endif()

if(USE_MLX5DV)
  message(STATUS "Enabled USE_MLX5DV support (mlx5 direct verbs)")
  target_link_libraries(transfer_engine PUBLIC mlx5)
endif()

if(USE_EFA)
  message(STATUS "Enabled USE_EFA (AWS Elastic Fabric Adapter) support")
  target_link_libraries(transfer_engine PUBLIC fabric efa_transport)
endif()
if(USE_CXI)
  message(STATUS "Enabled USE_CXI (HPE Cray Slingshot) support")
  target_link_libraries(transfer_engine PUBLIC fabric cxi_transport)
endif()

if(USE_UB)
  message(STATUS "Enabled USE_UB protocol support")
  target_link_libraries(transfer_engine PUBLIC ub_transport)
endif()
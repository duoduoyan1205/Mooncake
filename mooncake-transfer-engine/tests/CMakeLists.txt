if(USE_MEMORY_POOL)
  #
  # MPU external DMA-BUF helper programs
  #
  find_library(AMDGPU_LIBRARY
               NAMES amdgpu
               REQUIRED)

  find_library(AMDGPU_MPU_LIBRARY
               NAMES amdgpu-mpu
               REQUIRED)

  find_library(IBVERBS_LIBRARY
               NAMES ibverbs
               REQUIRED)

  add_executable(gpu_dmabuf_helper
                 ${WORKSPACE}/helpers/gpu_dmabuf_helper.c)

  target_link_libraries(gpu_dmabuf_helper
                        PRIVATE
                        ${AMDGPU_LIBRARY}
                        drm)

  add_executable(nic_dmabuf_helper
                 ${WORKSPACE}/helpers/nic_dmabuf_helper.c)

  target_link_libraries(nic_dmabuf_helper
                        PRIVATE
                        ${IBVERBS_LIBRARY})

  #
  # MPU external I/O integration test
  #
  add_executable(memory_pool_external_io_test
                 ${WORKSPACE}/memory_pool_external_io_test.cpp)

  target_link_libraries(memory_pool_external_io_test
                        PRIVATE
                        transfer_engine
                        ${AMDGPU_MPU_LIBRARY}
                        gtest
                        gtest_main)

  add_test(NAME memory_pool_external_io_test
           COMMAND memory_pool_external_io_test)

  set_tests_properties(memory_pool_external_io_test
                       PROPERTIES
                       LABELS "memory_pool;mpu;sue;hardware")
endif()

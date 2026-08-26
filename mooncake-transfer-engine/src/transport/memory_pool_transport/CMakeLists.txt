add_library(memory_pool_transport OBJECT memory_pool_transport.cpp)

target_include_directories(memory_pool_transport PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)

target_link_libraries(memory_pool_transport PRIVATE
  glog::glog
  pthread)

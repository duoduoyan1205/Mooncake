#include "memory_pool_storage_backend.h"
#include "storage_backend.h"

namespace mooncake {

tl::expected<std::shared_ptr<StorageBackendInterface>, ErrorCode>
CreateMemoryPoolStorageBackend(const FileStorageConfig& config) {
    return std::make_shared<MemoryPoolStorageBackend>(config);
}

}  // namespace mooncake

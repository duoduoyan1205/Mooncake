#include "transfer_metadata.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <memory>
#include <string>

#include <gtest/gtest.h>

namespace mooncake {
namespace {

uint16_t findAvailableTcpPort(int& sockfd) {
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) return 0;

    int reuse = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(sockfd);
        sockfd = -1;
        return 0;
    }
    if (listen(sockfd, 16) < 0) {
        close(sockfd);
        sockfd = -1;
        return 0;
    }

    socklen_t len = sizeof(addr);
    if (getsockname(sockfd, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
        close(sockfd);
        sockfd = -1;
        return 0;
    }
    return ntohs(addr.sin_port);
}

std::shared_ptr<TransferMetadata::SegmentDesc> makeMemoryPoolSegmentDesc(
    const std::string& name, uint64_t target_addr, uint64_t length) {
    auto segment_desc = std::make_shared<TransferMetadata::SegmentDesc>();
    segment_desc->name = name;
    segment_desc->protocol = "memory_pool";
    segment_desc->tcp_data_port = 0;

    TransferMetadata::BufferDesc buffer_desc;
    buffer_desc.name = "memory_pool_buffer";
    buffer_desc.length = length;
    buffer_desc.target_addr = target_addr;
    segment_desc->buffers.push_back(buffer_desc);
    return segment_desc;
}

}  // namespace

TEST(TransferMetadataTest, MemoryPoolTargetAddrRoundTrip) {
    constexpr uint64_t kTargetAddr = 0x123456789abc0000ULL;
    constexpr uint64_t kLength = 2ULL * 1024 * 1024;

    TransferMetadata server(P2PHANDSHAKE);
    TransferMetadata client(P2PHANDSHAKE);

    int sockfd = -1;
    const uint16_t port = findAvailableTcpPort(sockfd);
    ASSERT_GT(port, 0);
    const std::string remote_segment_name = "127.0.0.1:" + std::to_string(port);

    ASSERT_EQ(server.addLocalSegment(
                  LOCAL_SEGMENT_ID, remote_segment_name,
                  makeMemoryPoolSegmentDesc(remote_segment_name, kTargetAddr,
                                             kLength)),
              0);

    TransferMetadata::RpcMetaDesc rpc_desc;
    rpc_desc.ip_or_host_name = "127.0.0.1";
    rpc_desc.rpc_port = port;
    rpc_desc.sockfd = sockfd;
    ASSERT_EQ(server.addRpcMetaEntry(remote_segment_name, rpc_desc), 0);

    ASSERT_EQ(client.addLocalSegment(
                  LOCAL_SEGMENT_ID, "127.0.0.1:0",
                  makeMemoryPoolSegmentDesc("127.0.0.1:0", 0x3000, kLength)),
              0);

    const auto segment_id = client.getSegmentID(remote_segment_name);
    ASSERT_NE(segment_id, static_cast<TransferMetadata::SegmentID>(-1));

    auto remote_desc = client.getSegmentDescByID(segment_id);
    ASSERT_TRUE(remote_desc);
    ASSERT_EQ(remote_desc->protocol, "memory_pool");
    ASSERT_EQ(remote_desc->buffers.size(), 1u);
    ASSERT_EQ(remote_desc->buffers[0].length, kLength);
    ASSERT_EQ(remote_desc->buffers[0].target_addr, kTargetAddr);

    ASSERT_EQ(server.removeSegmentDesc(remote_segment_name), 0);
}

}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

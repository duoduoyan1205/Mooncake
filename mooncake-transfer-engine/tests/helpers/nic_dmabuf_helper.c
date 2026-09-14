/* SPDX-License-Identifier: MIT
 * Real RDMA DMA-BUF importer + RC QP bootstrap + one-sided RDMA READ/WRITE.
 * Control plane: TCP exchange. One instance sets MPU_NIC_CTRL_LISTEN_PORT;
 * the other sets MPU_NIC_CTRL_PEER_HOST and MPU_NIC_CTRL_PEER_PORT.
 * ABI: nic_dmabuf_helper <write|read> <size> <dma-buf-fd> <pattern>
 */
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAGIC 0x4d50554eU
#define VERSION 1U
#define IOVA 0x100000000ULL
#define TIMEOUT_MS 10000

struct ep {
    uint32_t magic, version, qpn, psn, gid_index, rkey;
    uint64_t iova, len;
    uint8_t gid[16];
};

static struct ibv_device *find_dev(const char *n)
{
    int x = 0;
    struct ibv_device **d = ibv_get_device_list(&x), *r = 0;

    for (int i = 0; d && i < x; i++)
        if (!strcmp(ibv_get_device_name(d[i]), n)) {
            r = d[i];
            break;
        }

    ibv_free_device_list(d);
    return r;
}

static int xsend(int f, const void *p, size_t n)
{
    const uint8_t *b = p;

    while (n) {
        ssize_t x = send(f, b, n, MSG_NOSIGNAL);

        if (x <= 0)
            return -errno;
        b += x;
        n -= x;
    }

    return 0;
}

static int xrecv(int f, void *p, size_t n)
{
    uint8_t *b = p;

    while (n) {
        ssize_t x = recv(f, b, n, MSG_WAITALL);

        if (x <= 0)
            return -errno;
        b += x;
        n -= x;
    }

    return 0;
}

static int listen_ctrl(void)
{
    const char *s = getenv("MPU_NIC_CTRL_LISTEN_PORT");
    int l, one = 1, c, e;
    struct sockaddr_in a = {0};

    if (!s)
        return -EINVAL;

    l = socket(AF_INET, SOCK_STREAM, 0);
    if (l < 0)
        return -errno;

    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t)atoi(s));

    setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (bind(l, (struct sockaddr *)&a, sizeof(a)) < 0 || listen(l, 1) < 0) {
        e = errno;
        close(l);
        return -e;
    }

    c = accept(l, 0, 0);
    e = errno;
    close(l);

    return c < 0 ? -e : c;
}

static int connect_ctrl(void)
{
    const char *h = getenv("MPU_NIC_CTRL_PEER_HOST");
    const char *s = getenv("MPU_NIC_CTRL_PEER_PORT");
    int f;
    struct sockaddr_in a = {0};

    if (!h || !s)
        return -EINVAL;

    f = socket(AF_INET, SOCK_STREAM, 0);
    if (f < 0)
        return -errno;

    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)atoi(s));

    if (inet_pton(AF_INET, h, &a.sin_addr) != 1 ||
        connect(f, (struct sockaddr *)&a, sizeof(a)) < 0) {
        int e = errno;

        close(f);
        return -e;
    }

    return f;
}

static int qi(struct ibv_qp *q, int p)
{
    struct ibv_qp_attr a = {0};

    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = p;
    a.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

    return ibv_modify_qp(q, &a,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                         IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

static int qrtr(struct ibv_qp *q, int p, int mtu,
                const struct ep *r, int sg)
{
    struct ibv_qp_attr a = {0};

    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = mtu;
    a.dest_qp_num = r->qpn;
    a.rq_psn = r->psn;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1;
    a.ah_attr.port_num = p;
    a.ah_attr.grh.sgid_index = sg;
    a.ah_attr.grh.hop_limit = 1;
    memcpy(&a.ah_attr.grh.dgid, r->gid, 16);

    return ibv_modify_qp(q, &a,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                         IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                         IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static int qrts(struct ibv_qp *q, uint32_t psn)
{
    struct ibv_qp_attr a = {0};

    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = psn;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.max_rd_atomic = 1;

    return ibv_modify_qp(q, &a,
                         IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                         IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                         IBV_QP_MAX_QP_RD_ATOMIC);
}

static int waitcq(struct ibv_cq *cq)
{
    for (int i = 0; i < TIMEOUT_MS; i++) {
        struct ibv_wc w;
        int n = ibv_poll_cq(cq, 1, &w);

        if (n < 0)
            return -EIO;
        if (n) {
            if (w.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "completion: %s\n",
                        ibv_wc_status_str(w.status));
                return -EIO;
            }
            return 0;
        }
        usleep(1000);
    }

    return -ETIMEDOUT;
}

static int checkbuf(int f, size_t n, unsigned p, int fill)
{
    uint8_t *b = mmap(0, n, PROT_READ | PROT_WRITE, MAP_SHARED, f, 0);

    if (b == MAP_FAILED)
        return -errno;

    if (fill)
        memset(b, p, n);
    else
        for (size_t i = 0; i < n; i++)
            if (b[i] != (uint8_t)p) {
                fprintf(stderr, "verify failed at %zu: %02x\n", i, b[i]);
                munmap(b, n);
                return -EIO;
            }

    munmap(b, n);
    return 0;
}

int main(int ac, char **av)
{
    if (ac != 5)
        return 77;

    const char *op = av[1], *dn = getenv("MPU_NIC_IB_DEVICE");
    size_t n = strtoull(av[2], 0, 0);
    int db = atoi(av[3]), port = 1, gi = 0, ctrl = -1, ret = 77;
    unsigned pat = strtoul(av[4], 0, 0) & 255;

    if (!dn || !n || (strcmp(op, "write") && strcmp(op, "read")))
        return 77;
    if (getenv("MPU_NIC_IB_PORT"))
        port = atoi(getenv("MPU_NIC_IB_PORT"));
    if (getenv("MPU_NIC_IB_GID_INDEX"))
        gi = atoi(getenv("MPU_NIC_IB_GID_INDEX"));

    struct ibv_device *d = find_dev(dn);
    struct ibv_context *c = 0;
    struct ibv_pd *pd = 0;
    struct ibv_mr *mr = 0;
    struct ibv_cq *cq = 0;
    struct ibv_qp *q = 0;
    struct ibv_port_attr pa;
    union ibv_gid gid;
    struct ep l = {0}, r = {0};

    if (!d || (c = ibv_open_device(d)) == 0)
        goto out;
    if (ibv_query_port(c, port, &pa) || pa.state != IBV_PORT_ACTIVE ||
        ibv_query_gid(c, port, gi, &gid))
        goto out;
    if (!(pd = ibv_alloc_pd(c)))
        goto out;

    mr = ibv_reg_dmabuf_mr(pd, 0, n, IOVA, db,
                           IBV_ACCESS_LOCAL_WRITE |
                           IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE);
    if (!mr) {
        fprintf(stderr, "ibv_reg_dmabuf_mr: %s\n", strerror(errno));
        goto out;
    }

    if (checkbuf(db, n, pat, !strcmp(op, "write")))
        goto out;

    if (!(cq = ibv_create_cq(c, 2, 0, 0, 0)))
        goto out;

    struct ibv_qp_init_attr qa = {0};

    qa.send_cq = cq;
    qa.recv_cq = cq;
    qa.cap.max_send_wr = 1;
    qa.cap.max_recv_wr = 1;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    qa.qp_type = IBV_QPT_RC;

    q = ibv_create_qp(pd, &qa);
    if (!q || qi(q, port))
        goto out;

    l.magic = MAGIC;
    l.version = VERSION;
    l.qpn = q->qp_num;
    l.psn = (uint32_t)getpid() & 0xffffff;
    l.gid_index = gi;
    l.rkey = mr->rkey;
    l.iova = IOVA;
    l.len = n;
    memcpy(l.gid, gid.raw, 16);

    ctrl = getenv("MPU_NIC_CTRL_LISTEN_PORT") ?
           listen_ctrl() : connect_ctrl();
    if (ctrl < 0 || xsend(ctrl, &l, sizeof(l)) ||
        xrecv(ctrl, &r, sizeof(r)))
        goto out;
    if (r.magic != MAGIC || r.version != VERSION || r.len != n)
        goto out;

    if (qrtr(q, port, pa.active_mtu, &r, gi) || qrts(q, l.psn))
        goto out;

    struct ibv_sge s = {
        .addr = IOVA,
        .length = (uint32_t)n,
        .lkey = mr->lkey,
    };
    struct ibv_send_wr w = {0}, *bad = 0;

    w.wr_id = 1;
    w.sg_list = &s;
    w.num_sge = 1;
    w.opcode = !strcmp(op, "write") ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
    w.send_flags = IBV_SEND_SIGNALED;
    w.wr.rdma.remote_addr = r.iova;
    w.wr.rdma.rkey = r.rkey;

    if (ibv_post_send(q, &w, &bad) || waitcq(cq))
        goto out;
    if (!strcmp(op, "read") && checkbuf(db, n, pat, 0))
        goto out;

    printf("NIC RDMA %s PASS: %zu bytes local_qpn=%u remote_qpn=%u\n",
           op, n, l.qpn, r.qpn);
    ret = 0;

out:
    if (ctrl >= 0)
        close(ctrl);
    if (q)
        ibv_destroy_qp(q);
    if (cq)
        ibv_destroy_cq(cq);
    if (mr)
        ibv_dereg_mr(mr);
    if (pd)
        ibv_dealloc_pd(pd);
    if (c)
        ibv_close_device(c);
    return ret;
}

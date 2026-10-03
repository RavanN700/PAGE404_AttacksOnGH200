#ifndef __UVM_CPU_PAGE_CACHE_H__
#define __UVM_CPU_PAGE_CACHE_H__

#include "uvm_common.h"
#include <linux/hashtable.h>
#include <linux/spinlock.h>
#include <linux/mm.h>

// ── Module parameter ─────────────────────────────────────────────────────────
extern int uvm_cpu_page_cache_enable;

// ── Debug macro ──────────────────────────────────────────────────────────────
// Prints only when the cache is enabled, prefixed with [UVM_CACHE] for easy
// grep: sudo dmesg | grep "\[UVM_CACHE\]"
#define UVM_CACHE_DBG(fmt, ...)                                             \
    do {                                                                    \
        if (uvm_cpu_page_cache_enable)                                      \
            pr_info("[UVM_CACHE] %s:%d " fmt "\n",                         \
                    __func__, __LINE__, ##__VA_ARGS__);                     \
    } while (0)

// RC tracking macro — logs page refcount before and after operations
#define UVM_CACHE_RC(label, page, fmt, ...)                                 \
    do {                                                                    \
        if (uvm_cpu_page_cache_enable)                                      \
            pr_info("[UVM_CACHE_RC] %s:%d pfn=%lu rc=%d " label " " fmt "\n", \
                    __func__, __LINE__,                                     \
                    page_to_pfn(page), page_count(page),                   \
                    ##__VA_ARGS__);                                         \
    } while (0)

#define UVM_CPU_PAGE_CACHE_BITS 12

struct uvm_cpu_page_cache_entry {
    NvU64              vaddr;
    struct page       *page;
    bool               ref_held;
    bool               pending_gpu_migration; 
    struct hlist_node  node;
};

struct uvm_cpu_page_cache {
    DECLARE_HASHTABLE(table, UVM_CPU_PAGE_CACHE_BITS);
    spinlock_t lock;
};

void           uvm_cpu_page_cache_init(struct uvm_cpu_page_cache *cache);
struct page   *uvm_cpu_page_cache_lookup(struct uvm_cpu_page_cache *cache, NvU64 vaddr);
void           uvm_cpu_page_cache_insert(struct uvm_cpu_page_cache *cache, NvU64 vaddr, struct page *page);
struct page   *uvm_cpu_page_cache_take(struct uvm_cpu_page_cache *cache, NvU64 vaddr);
void           uvm_cpu_page_cache_destroy(struct uvm_cpu_page_cache *cache);
bool           uvm_cpu_page_cache_drop_ref(struct uvm_cpu_page_cache *cache, NvU64 vaddr);
void           uvm_cpu_page_cache_restore_ref(struct uvm_cpu_page_cache *cache, NvU64 vaddr);
void           uvm_cpu_page_cache_clear_pending(struct uvm_cpu_page_cache *cache, NvU64 vaddr);
void           uvm_cpu_page_cache_restore_ref_if_pending(struct uvm_cpu_page_cache *cache, NvU64 vaddr);
//void           uvm_cpu_page_cache_restore_ref_unconditional(struct uvm_cpu_page_cache *cache, NvU64 vaddr);
#endif /* __UVM_CPU_PAGE_CACHE_H__ */
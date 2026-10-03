#include "uvm_cpu_page_cache.h"
#include <linux/hashtable.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/spinlock.h>
#include <linux/moduleparam.h>   // ← ADD

// Module parameter — default off, set with modprobe nvidia-uvm uvm_cpu_page_cache_enable=1
int uvm_cpu_page_cache_enable = 0;
module_param(uvm_cpu_page_cache_enable, int, 0444);
MODULE_PARM_DESC(uvm_cpu_page_cache_enable,
    "Enable CPU page cache to preserve physical addresses across GPU migrations (0=off, 1=on)");

void uvm_cpu_page_cache_init(struct uvm_cpu_page_cache *cache)
{
    hash_init(cache->table);
    spin_lock_init(&cache->lock);
    UVM_CACHE_DBG("cache initialized at %px", cache);
}

void uvm_cpu_page_cache_insert(struct uvm_cpu_page_cache *cache,
                                NvU64 vaddr, struct page *page)
{
    struct uvm_cpu_page_cache_entry *entry;
    unsigned long flags;

    UVM_ASSERT(!uvm_cpu_page_cache_lookup(cache, vaddr));

    entry = kmalloc(sizeof(*entry), GFP_ATOMIC);
    if (!entry) {
        UVM_CACHE_DBG("kmalloc failed for vaddr=0x%llx — cache miss will persist", vaddr);
        return;
    }

    entry->vaddr = vaddr;
    entry->page  = page;
    entry->ref_held = true; 
    entry->pending_gpu_migration = false;
    INIT_HLIST_NODE(&entry->node);

    UVM_CACHE_RC("BEFORE INSERT", page, "vaddr=0x%llx", vaddr);
    get_page(page);   // permanent cache ref
    UVM_CACHE_RC("AFTER INSERT", page, "vaddr=0x%llx", vaddr);

    UVM_CACHE_DBG("INSERT vaddr=0x%llx page=%px PA=0x%llx pfn=%lu rc_after=%d",
                  vaddr, page,
                  (unsigned long long)page_to_phys(page),
                  page_to_pfn(page),
                  page_count(page));

    spin_lock_irqsave(&cache->lock, flags);
    hash_add(cache->table, &entry->node, vaddr);
    spin_unlock_irqrestore(&cache->lock, flags);
}

struct page *uvm_cpu_page_cache_lookup(struct uvm_cpu_page_cache *cache,
                                        NvU64 vaddr)
{
    struct uvm_cpu_page_cache_entry *entry;
    struct page *found = NULL;
    unsigned long flags;

    spin_lock_irqsave(&cache->lock, flags);
    hash_for_each_possible(cache->table, entry, node, vaddr) {
        if (entry->vaddr == vaddr) {
            found = entry->page;
            break;
        }
    }
    spin_unlock_irqrestore(&cache->lock, flags);

    if (found)
        UVM_CACHE_DBG("LOOKUP HIT  vaddr=0x%llx page=%px PA=0x%llx pfn=%lu",
                      vaddr, found,
                      (unsigned long long)page_to_phys(found),
                      page_to_pfn(found));
    else
        UVM_CACHE_DBG("LOOKUP MISS vaddr=0x%llx", vaddr);

    return found;
}

struct page *uvm_cpu_page_cache_take(struct uvm_cpu_page_cache *cache,
                                      NvU64 vaddr)
{
    struct uvm_cpu_page_cache_entry *entry;
    struct page *found = NULL;
    unsigned long flags;

    spin_lock_irqsave(&cache->lock, flags);
    hash_for_each_possible(cache->table, entry, node, vaddr) {
        if (entry->vaddr == vaddr) {
            found = entry->page;
            hash_del(&entry->node);
            kfree(entry);
            break;
        }
    }
    spin_unlock_irqrestore(&cache->lock, flags);

    if (found)
        UVM_CACHE_DBG("TAKE vaddr=0x%llx page=%px PA=0x%llx pfn=%lu rc_after=%d",
                      vaddr, found,
                      (unsigned long long)page_to_phys(found),
                      page_to_pfn(found),
                      page_count(found));
    else
        UVM_CACHE_DBG("TAKE MISS vaddr=0x%llx", vaddr);

    return found;
}

void uvm_cpu_page_cache_destroy(struct uvm_cpu_page_cache *cache)
{
    struct uvm_cpu_page_cache_entry *entry;
    struct hlist_node *tmp;
    unsigned int bkt;
    int count = 0;

    hash_for_each_safe(cache->table, bkt, tmp, entry, node) {
        UVM_CACHE_DBG("DESTROY freeing vaddr=0x%llx PA=0x%llx ref_held=%d rc=%d",
                      entry->vaddr,
                      (unsigned long long)page_to_phys(entry->page),
                      entry->ref_held,
                      page_count(entry->page));
        hash_del(&entry->node);
        if (entry->ref_held && page_count(entry->page) > 0) {
            UVM_CACHE_RC("BEFORE DESTROY put", entry->page, "vaddr=0x%llx", entry->vaddr);
            /* int nid = page_to_nid(entry->page);
            if (!node_state(nid, N_CPU)) {
                pr_err("[UVM_CACHE] BUG: GPU page in cache at destroy! "
                    "pfn=%lu nid=%d PA=0x%llx vaddr=0x%llx -- SKIPPING put_page\n",
                    page_to_pfn(entry->page), nid,
                    (unsigned long long)page_to_phys(entry->page),
                    entry->vaddr);
                // Remove from hash but DO NOT put_page
                kfree(entry);
                continue;
            }  */
            //put_page(entry->page);
        } else if (entry->ref_held) {
            UVM_CACHE_DBG("DESTROY skipped put — rc already 0 vaddr=0x%llx", entry->vaddr);
        }
        kfree(entry);
        count++;
    }

    UVM_CACHE_DBG("DESTROY done — released %d entries from cache %px", count, cache);
}

// In uvm_cpu_page_cache.c:
bool uvm_cpu_page_cache_drop_ref(struct uvm_cpu_page_cache *cache, NvU64 vaddr)
{
    struct uvm_cpu_page_cache_entry *entry;
    bool dropped = false;
    unsigned long flags;

    spin_lock_irqsave(&cache->lock, flags);
    hash_for_each_possible(cache->table, entry, node, vaddr) {
        if (entry->vaddr == vaddr) {
            if (entry->ref_held) {
                if (page_count(entry->page) > 1) {
                    // Normal case: rc=2, drop to 1 for isolation
                    UVM_CACHE_RC("BEFORE DROP", entry->page, "vaddr=0x%llx", vaddr);
                    put_page(entry->page);
                    UVM_CACHE_RC("AFTER DROP", entry->page, "vaddr=0x%llx", vaddr);
                    UVM_CACHE_DBG("DROP actual put vaddr=0x%llx rc_now=%d",
                                  vaddr, page_count(entry->page));
                } else {
                    // rc=1 already (after HIT migration), don't put to 0
                    // Just mark ref_held=false so PRESERVE adds get_page
                    // before migrate_vma_finalize frees src page
                    UVM_CACHE_DBG("DROP skip put (rc=1) vaddr=0x%llx", vaddr);
                }
                // Either way: mark not held so PRESERVE will call get_page
                entry->ref_held = false;
                entry->pending_gpu_migration = true;
                dropped = true;
            }
            break;
        }
    }
    spin_unlock_irqrestore(&cache->lock, flags);
    return dropped;
}

void uvm_cpu_page_cache_restore_ref(struct uvm_cpu_page_cache *cache, NvU64 vaddr)
{
    struct uvm_cpu_page_cache_entry *entry;
    unsigned long flags;

    spin_lock_irqsave(&cache->lock, flags);
    hash_for_each_possible(cache->table, entry, node, vaddr) {
        if (entry->vaddr == vaddr) {
            // Always get_page — migrate_vma_finalize always puts src page
            // regardless of whether CPU migration raced with us
            UVM_CACHE_RC("BEFORE RESTORE", entry->page, "vaddr=0x%llx", vaddr);
            get_page(entry->page);
            UVM_CACHE_RC("AFTER RESTORE", entry->page, "vaddr=0x%llx", vaddr); 
            entry->ref_held = true;
            entry->pending_gpu_migration = false;
            UVM_CACHE_DBG("RESTORE vaddr=0x%llx rc_now=%d",
                          vaddr, page_count(entry->page));
            break;
        }
    }
    spin_unlock_irqrestore(&cache->lock, flags);
}

void uvm_cpu_page_cache_clear_pending(struct uvm_cpu_page_cache *cache, NvU64 vaddr)
{
    struct uvm_cpu_page_cache_entry *entry;
    unsigned long flags;

    spin_lock_irqsave(&cache->lock, flags);
    hash_for_each_possible(cache->table, entry, node, vaddr) {
        if (entry->vaddr == vaddr) {
            entry->pending_gpu_migration = false;
            entry->ref_held = true;   // ← claim existing rc as cache's ref (no get_page)
            break;
        }
    }
    spin_unlock_irqrestore(&cache->lock, flags);
}

void uvm_cpu_page_cache_restore_ref_if_pending(
    struct uvm_cpu_page_cache *cache, NvU64 vaddr)
{
    struct uvm_cpu_page_cache_entry *entry;
    unsigned long flags;

    spin_lock_irqsave(&cache->lock, flags);
    hash_for_each_possible(cache->table, entry, node, vaddr) {
        if (entry->vaddr == vaddr) {
            if (!entry->ref_held && entry->pending_gpu_migration) {
                // Migration failed — finalize did NOT put_page
                // so do NOT get_page — just reclaim ref_held for existing rc=1
                UVM_CACHE_RC("BEFORE RESTORE_IF_PENDING", entry->page, "vaddr=0x%llx", vaddr);
                get_page(entry->page);
                UVM_CACHE_RC("AFTER RESTORE_IF_PENDING", entry->page, "vaddr=0x%llx", vaddr);
                entry->ref_held = true;
                entry->pending_gpu_migration = false;
                UVM_CACHE_DBG("RESTORE_FAILED_MIG vaddr=0x%llx rc_now=%d",
                              vaddr, page_count(entry->page));
            }
            break;
        }
    }
    spin_unlock_irqrestore(&cache->lock, flags);
}

/* void uvm_cpu_page_cache_restore_ref_unconditional(
    struct uvm_cpu_page_cache *cache, NvU64 vaddr)
{
    struct uvm_cpu_page_cache_entry *entry;
    unsigned long flags;

    spin_lock_irqsave(&cache->lock, flags);
    hash_for_each_possible(cache->table, entry, node, vaddr) {
        if (entry->vaddr == vaddr) {
            if (!entry->ref_held) {
                get_page(entry->page);
                entry->ref_held = true;
                entry->pending_gpu_migration = false;
                UVM_CACHE_DBG("RESTORE_UNCONDITIONAL vaddr=0x%llx rc_now=%d",
                              vaddr, page_count(entry->page));
            }
            break;
        }
    }
    spin_unlock_irqrestore(&cache->lock, flags);
} */
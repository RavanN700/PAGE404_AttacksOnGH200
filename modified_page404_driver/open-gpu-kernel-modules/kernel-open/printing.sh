# Find the actual enum definition
grep -rn "uvm_make_resident_cause_t\|UVM_MAKE_RESIDENT_CAUSE" \
    ~/Driver/open-gpu-kernel-modules/kernel-open/nvidia-uvm/uvm_va_block.h | head -20

# Also search other headers
grep -rn "typedef.*make_resident_cause\|UVM_MAKE_RESIDENT_CAUSE_API\|UVM_MAKE_RESIDENT_CAUSE_PREFETCH" \
    ~/Driver/open-gpu-kernel-modules/kernel-open/nvidia-uvm/*.h | grep -v "^Binary"

grep -rn "UVM_MAKE_RESIDENT_CAUSE\|make_resident_cause" \
    ~/Driver/open-gpu-kernel-modules/kernel-open/nvidia-uvm/uvm_migrate.c | head -20
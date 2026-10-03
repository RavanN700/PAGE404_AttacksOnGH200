sudo modprobe -r nvidia-uvm
sudo modprobe nvidia-uvm uvm_cpu_page_cache_enable=1
cat /sys/module/nvidia_uvm/parameters/uvm_cpu_page_cache_enable
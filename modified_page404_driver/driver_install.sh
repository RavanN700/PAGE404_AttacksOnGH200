cd ./open-gpu-kernel-modules
make modules -j$(nproc)
sudo make modules_install -j$(nproc)
sudo sh ./NVIDIA-Linux-aarch64-590.48.01.run --no-kernel-modules
sudo nvidia-uninstall
sudo apt purge -y '^nvidia-*' '^libnvidia-*'
sudo rm -r /var/lib/dkms/nvidia
sudo apt -y autoremove
sudo update-initramfs -c -k `uname -r`
sudo update-grub2

sudo rm -f /usr/lib/modules/$(uname -r)/extra/nvidia*.ko
sudo rm -f /etc/modprobe.d/nvidia*.conf
sudo rm -f /usr/lib/x86_64-linux-gnu/libcuda*
sudo rm -f /usr/lib/x86_64-linux-gnu/libnv*
sudo depmod -a
sudo update-initramfs -u -k $(uname -r)

read -p "Press any key to reboot... " -n1 -s
sudo reboot

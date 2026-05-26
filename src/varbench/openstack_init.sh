#cloud-config
users:
  - name: ubuntu
    plain_text_passwd: 'YourSecurePasswordHere'
    lock_passwd: false
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    ssh-authorized-keys:
      - ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD... your-public-key

chpasswd:
  list: |
    ubuntu:YourSecurePasswordHere
  expire: False

ssh_pwauth: true


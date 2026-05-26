# Download kind to your local directory (adjust the version as needed)
curl -Lo ~/kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64

# Make it executable
chmod +x ~/kind

# Optionally, add kind to your PATH for easier access
echo 'export PATH=$PATH:$HOME' >> ~/.bashrc
source ~/.bashrc


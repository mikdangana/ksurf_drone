export APPTAINER_CACHEDIR=/project/def-PROJECT_OWNER/OS_USERNAME

apptainer cache clean --force 
apptainer build --fakeroot --ignore-fakeroot-command k3s_container.sif k3s.def

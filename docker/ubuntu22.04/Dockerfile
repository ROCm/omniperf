FROM --platform=linux/amd64 ubuntu:22.04
WORKDIR /home

USER root

ENV DEBIAN_FRONTEND noninteractive
ENV TZ "US/Chicago"

RUN apt-get update
RUN apt-get install -y wget gpg

RUN  mkdir --parents --mode=0755 /etc/apt/keyrings

# Download the key, convert the signing-key to a full
# keyring required by apt and store in the keyring directory
RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

# amdgpu repository for jammy
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/5.6/ubuntu jammy main" \
    | tee /etc/apt/sources.list.d/amdgpu.list
RUN apt update
RUN apt install -y amdgpu-dkms

# ROCm repositories for jammy
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/5.6 jammy main" \
    | tee --append /etc/apt/sources.list.d/rocm.list

RUN echo "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" | tee /etc/apt/preferences.d/rocm-pin-600
RUN apt update
RUN apt install -y rocm-hip-sdk
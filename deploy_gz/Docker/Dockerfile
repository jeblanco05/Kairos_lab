# The robotnik_base named build context is supplied by docker-compose.yml.
FROM robotnik_base

ARG ONNXRUNTIME_VERSION=1.22.0

USER root

# Controlador mecanum que usa el perfil de RB-Kairos (no viene en ros-jazzy-desktop).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ros-jazzy-mecanum-drive-controller \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime's C/C++ SDK for kairos_deploy (amd64).
RUN curl -fsSL --retry 3 \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
        -o /tmp/onnxruntime.tgz \
    && mkdir -p /tmp/onnxruntime /opt/onnxruntime \
    && tar -xzf /tmp/onnxruntime.tgz -C /tmp/onnxruntime --strip-components=1 \
    && cp -a /tmp/onnxruntime/include /opt/onnxruntime/ \
    && cp -a /tmp/onnxruntime/lib /opt/onnxruntime/ \
    && ln -s . /opt/onnxruntime/include/onnxruntime \
    && ln -s /opt/onnxruntime/lib /opt/onnxruntime/lib64 \
    && echo /opt/onnxruntime/lib > /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig \
    && rm -rf /tmp/onnxruntime /tmp/onnxruntime.tgz

USER robot
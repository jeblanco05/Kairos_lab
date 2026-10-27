FROM nvcr.io/nvidia/isaac-sim:4.5.0

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color 

# Recibir el UID y GID del host al construir para replicar permisos
ARG UID=1000
ARG GID=1000

# Crear un usuario no-root llamado 'devuser' (y darle sudo por si lo necesitas)
RUN groupadd -g ${GID} hostgroup || true && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash devuser && \
    apt-get update && apt-get install -y sudo git nano && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clonar e instalar Isaac Lab v2.3.2
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    git checkout v2.3.2 && \
    ln -s /isaac-sim _isaac_sim && \
    find source -type f -name "setup.py" -exec sed -i 's/flatdict==4.0.1/flatdict==4.1.0/g' {} + && \
    ./isaaclab.sh --install

# Cambiar la propiedad de todo /workspace al nuevo usuario
RUN chown -R devuser:${GID} /workspace && \
    mkdir -p /home/devuser/.local/share/ov /home/devuser/.nvidia-omniverse && \
    chown -R devuser:${GID} /home/devuser

# Cambiar al usuario normal para la ejecución
USER devuser

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

CMD ["sleep", "infinity"]
#!/bin/bash

# Asegurarnos de estar en el directorio correcto dentro del contenedor
cd ..

echo "Clonando repositorios..."

# Comprobar si rsl_rl ya existe para no clonarlo dos veces
if [ ! -d "rsl_rl" ]; then
    git clone https://github.com/jeblanco05/rsl_rl.git
    cd rsl_rl
    git checkout version-3.1.2
    
    # Si necesitas instalarlo (ej. pip install -e .), puedes añadir el comando aquí:
    # pip install -e .
    
    cd ..
    echo "rsl_rl clonado y listo."
else
    echo "El repositorio rsl_rl ya existe. Omitiendo..."
fi

# Aquí puedes añadir más repositorios en el futuro
# git clone https://github.com/otro/repo.git

echo "¡Instalación completada!"
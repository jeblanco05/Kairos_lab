#!/bin/bash
#SBATCH --job-name=Train-Kairos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --qos=regular
#SBATCH --output=logs_2/Train-Kairos_%j.log
#SBATCH --gres=gpu:L4:1

echo "Iniciando Train Kairos"

MY_RESEARCH_ROOT="/mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab"
UNITREE_SOURCE="/mnt/beegfs/home/jesuseliseo.blanco/my_projects/unitree_rl_lab/source/unitree_rl_lab"

export PYTHONPATH=$PYTHONPATH:$MY_RESEARCH_ROOT:$UNITREE_SOURCE
export OMNI_USERDATA_PATH="/mnt/beegfs/home/jesuseliseo.blanco/isaac_sim_cache/userdata"
export OMNI_CACHE_PATH="/mnt/beegfs/home/jesuseliseo.blanco/isaac_sim_cache/cache"
export OMNI_KIT_CACHE="/mnt/beegfs/home/jesuseliseo.blanco/isaac_sim_cache/kit_cache"
export GIT_PYTHON_REFRESH=quiet

mkdir -p $OMNI_USERDATA_PATH $OMNI_CACHE_PATH $OMNI_KIT_CACHE

# Ejecución con writable-tmpfs para permitir escritura en Isaac Sim (necesario para video)
# --num_envs 1 \
# --max_iterations 50 \
# --load_run my_projects/Kairos_lab/logs/Kairos-PPO-v1/ppo \
# --checkpoint -1 \
singularity exec --nv --writable-tmpfs \
  --bind /mnt/beegfs:/mnt/beegfs \
  --env OMNI_USERDATA_PATH=$OMNI_USERDATA_PATH \
  --env OMNI_CACHE_PATH=$OMNI_CACHE_PATH \
  --env OMNI_KIT_CACHE=$OMNI_KIT_CACHE \
  --env PYTHONPATH=$PYTHONPATH \
  --env GIT_PYTHON_REFRESH=quiet \
  /mnt/beegfs/home/jesuseliseo.blanco/isaac-sim-4.5.0.sif \
  /isaac-sim/python.sh -u my_projects/Kairos_lab/scripts/train.py \
  --task Kairos-PPO-v1 \
  --log_dir my_projects/Kairos_lab/logs \
  --num_envs 50 \
  --max_iterations 50 \
  --headless \
  --video \
  --video_length 200 \
  --video_interval 25000 \
  --enable_cameras

echo "Script finalizado, liberando recursos."
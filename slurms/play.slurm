#!/bin/bash
#SBATCH --job-name=Play-Kairos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH --output=logs_2/Play-Kairos_%j.log
#SBATCH --gres=gpu:L4:1
echo "Iniciando play Kairos"

MY_RESEARCH_ROOT="/mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab"
UNITREE_SOURCE="/mnt/beegfs/home/jesuseliseo.blanco/my_projects/unitree_rl_lab/source/unitree_rl_lab"

export PYTHONPATH=$PYTHONPATH:$MY_RESEARCH_ROOT:$UNITREE_SOURCE
export OMNI_USERDATA_PATH="/mnt/beegfs/home/jesuseliseo.blanco/isaac_sim_cache/userdata"
export OMNI_CACHE_PATH="/mnt/beegfs/home/jesuseliseo.blanco/isaac_sim_cache/cache"
export OMNI_KIT_CACHE="/mnt/beegfs/home/jesuseliseo.blanco/isaac_sim_cache/kit_cache"
export GIT_PYTHON_REFRESH=quiet

mkdir -p $OMNI_USERDATA_PATH $OMNI_CACHE_PATH $OMNI_KIT_CACHE

# task & load_run
singularity exec --nv --writable-tmpfs \
  --bind /mnt/beegfs:/mnt/beegfs \
  --env OMNI_USERDATA_PATH=$OMNI_USERDATA_PATH \
  --env OMNI_CACHE_PATH=$OMNI_CACHE_PATH \
  --env OMNI_KIT_CACHE=$OMNI_KIT_CACHE \
  --env PYTHONPATH=$PYTHONPATH \
  --env GIT_PYTHON_REFRESH=quiet \
  /mnt/beegfs/home/jesuseliseo.blanco/isaac-sim-4.5.0.sif \
  /isaac-sim/python.sh -u my_projects/Kairos_lab/scripts/play_.py \
  --task Kairos-PPO-v1 \
  --load_run /mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/logs/Kairos-PPO-v1 \
  --checkpoint -1 \
  --num_envs 25 \
  --headless \
  --video \
  --video_length 500 \
  --enable_cameras

echo "Script finalizado, liberando recursos."

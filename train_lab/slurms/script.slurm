#!/bin/bash
#SBATCH --job-name=Kairos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=6:00:00
#SBATCH --qos=regular
#SBATCH --output=logs_2/Kairos-G1_%j.log
#SBATCH --gres=gpu:L4:1

echo "Iniciando Train G1"

export PYTHONPATH=$PYTHONPATH:$MY_RESEARCH_ROOT:$UNITREE_SOURCE
export OMNI_USERDATA_PATH="/mnt/beegfs/home/jesuseliseo.blanco/isaac_sim_cache/userdata"
export OMNI_CACHE_PATH="/mnt/beegfs/home/jesuseliseo.blanco/isaac_sim_cache/cache"
export OMNI_KIT_CACHE="/mnt/beegfs/home/jesuseliseo.blanco/isaac_sim_cache/kit_cache"
export GIT_PYTHON_REFRESH=quiet

mkdir -p $OMNI_USERDATA_PATH $OMNI_CACHE_PATH $OMNI_KIT_CACHE

# Ejecución con writable-tmpfs para permitir escritura en Isaac Sim (necesario para video)

# /mnt/beegfs/home/jesuseliseo.blanco/IsaacLab/scripts/tools/convert_urdf.py \
# /mnt/beegfs/home/jesuseliseo.blanco/my_projects/GO2Sim/unitree_go2_ros2_jazzy/unitree_go2_description/urdf/go2_d1_robot.urdf \
# /mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/assets/GO2/go2_d1.usd

# /mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/scripts/import_Kairos.py --headless 
# fix_usd
# /mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/scripts/play.py --headless --record_video 

# /mnt/beegfs/home/jesuseliseo.blanco/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
#   --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --headless --video

# /isaac-sim/python.sh -u /mnt/beegfs/home/jesuseliseo.blanco/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
#   --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
#   --load_run 2026-10-07_14-06-09 \
#   --num_envs 64 --headless --video

#rm -f /dev/shm/sem.carbonite-sharedmemory
singularity exec --nv --writable-tmpfs \
  --bind /mnt/beegfs:/mnt/beegfs \
  --env OMNI_USERDATA_PATH=$OMNI_USERDATA_PATH \
  --env OMNI_CACHE_PATH=$OMNI_CACHE_PATH \
  --env OMNI_KIT_CACHE=$OMNI_KIT_CACHE \
  --env PYTHONPATH=$PYTHONPATH \
  --env GIT_PYTHON_REFRESH=quiet \
  /mnt/beegfs/home/jesuseliseo.blanco/isaac-sim-4.5.0.sif \
  /isaac-sim/python.sh -u /mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/scripts/check_train_env.py --headless --max_level 4 --stds 0
echo "Script finalizado, liberando recursos."
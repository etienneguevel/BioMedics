#!/bin/bash
#SBATCH --job-name=BioMedics_full_pipeline
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition gpuV100
#SBATCH --output=/export/home/cse200055/Etienne/ai_triomph/log/slurm-%x-%j-stdout.log
#SBATCH --error=/export/home/cse200055/Etienne/ai_triomph/log/slurm-%x-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable

source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
cd "/export/home/cse200055/Etienne/BioMedics"
source "env/bin/activate"
conda deactivate
echo "Python used: $(which python)"

start_time=$(date +%s)

echo "----------Starting the full pipeline----------"
python biomedics/run/full_pipeline.py --config_path biomedics/run/configs/config_full_pipeline.yaml
end_time=$(date +%s)
echo "----------Full pipeline done----------"

echo "Time taken: $((end_time - start_time)) seconds"

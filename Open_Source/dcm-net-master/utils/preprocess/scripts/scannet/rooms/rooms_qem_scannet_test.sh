
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=qem_rooms_test
#SBATCH --time=14-00:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=output/qem_rooms_test_%A_%a.txt
#SBATCH --array=0-99

cd ../../../

python3.7 graph_level_generation.py \
--in_path raw_data/scannet/scans_test \
--out_path data/scannet_qem_test_rooms/ \
--level_params 0.04 30 30 30 \
--test \
--qem \
--dataset scannet \
--number $((SLURM_ARRAY_TASK_ID))
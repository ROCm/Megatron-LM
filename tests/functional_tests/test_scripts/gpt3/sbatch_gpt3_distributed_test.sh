#!/bin/bash

# Parameters
#SBATCH --account=llmservice_dev_mcore
#SBATCH --job-name=llmservice_dev_mcore-ci:megatron-job
#SBATCH --nodes=1
#SBATCH --partition=luna

DATA_PATH=/workspace/data/gpt3_data/my-gpt3_00_text_document
CHECKPOINT_PATH=/workspace/checkpoints
TENSORBOARD_DIR=/workspace/logs

if [[ -n $MBS ]]; then MBS=4; fi
if [[ -n $GBS ]]; then GBS=32; fi

if [[ -n $VP_SIZE ]]; then VP_SIZE="" ; fi

echo 'Running tests using $PYTORCH_IMAGE image'

export DATA_PATH CHECKPOINT_PATH TENSORBOARD_DIR USE_TE TP_SIZE PP_SIZE NUM_NODES MAX_STEPS USE_CORE VP_SIZE MBS GBS ADDITIONAL_PARAMS

envsubst <$BUILD_DIR/tests/functional_tests/test_scripts/gpt3/sbatch_gpt3_distributed_test.sh > $BASE_DIR/scripts/sbatch_gpt3_distributed_test.sh

srun --output $BASE_DIR/results/slurm-%j.out --error $BASE_DIR/results/slurm-%j.out --container-image $PYTORCH_IMAGE --container-mounts $BASE_DIR/logs:/workspace/logs,$BASE_DIR/checkpoints:/workspace/checkpoints,$BUILD_DIR:/workspace/megatron-lm,$DATA_DIR:/workspace/data --no-container-mount-home bash -c "
  ls 
  cd /workspace/megatron-lm
  ./tests/functional_tests/test_scripts/gpt3/pretrain_gpt3_distributed_test.sh $DATA_PATH $CHECKPOINT_PATH $TENSORBOARD_DIR $USE_TE $TP_SIZE $PP_SIZE $NUM_NODES $MAX_STEPS $USE_CORE \"$VP_SIZE\" \"$MBS\" \"$GBS\" \"$ADDITIONAL_PARAMS\""

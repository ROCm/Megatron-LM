MBS=1

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TORCH_BLAS_PREFER_HIPBLASLT=0 TP=4 PP=1 MBS=$MBS BS=128 EPOCHS=1 bash finetune_llama2_chat.sh --no-torch-compile
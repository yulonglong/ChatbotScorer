optimistic_model_output_folder="expt0191-1-dIRISltmax-v4000-e100-tcnn-pmeanot-c300w3cl1-rlstm300rl2-armsprop-b16-seed1178-TITANX1"
pessimistic_model_output_folder="expt0222-1-dIRISltmin-v4000-e100-tcnn-pmeanot-c300w3cl1-rlstm300rl2-armsprop-b16-seed1278-TITANX2"
averaging_model_output_folder="expt1093-1-dIRISltmean-v4000-e100-tcnn-pmeanot-c300w3cl1-rlstm300rl2-armsprop-b16-seed1378-TITANX3"
output_voting_folder="voting"
python voting.py \
-maxp ${optimistic_model_output_folder}/preds/ \
-minp ${pessimistic_model_output_folder}/preds/ \
-ref ${averaging_model_output_folder}/preds/ \
-o ${output_voting_folder} \

#echo "IMDB"
CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.explain \
                    --ckpt results/imdb/latent_30pct \
                    --ckpt_comm results/comm_imdb/latent_30pct \
                    --save_path explanations/imdb_latent_30pct_fixl_dev

CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.explain \
                    --ckpt results/imdb/bernoulli_sparsity0003 \
                    --ckpt_comm  results/comm_imdb/bernoulli_sparsity0003 \
                    --save_path explanations/imdb_bernoulli_sparsity0003_fixl_dev

#echo "SNLI"
CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.snli.explain \
                    --ckpt results/snli/latent_10pct \
                    --ckpt_comm results/comm_snli/latent_10pct  \
                    --save_path explanations/snli_latent_10pct_fixl_dev

CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.snli.explain \
                    --ckpt results/snli/bernoulli_sparsity0003_fix_new_indep_qk \
                    --ckpt_comm results/comm_snli/bernoulli_sparsity0003_fix_new_indep_qk \
                    --save_path explanations/snli_bernoulli_sparsity0003_fixl_dev


cp explanations/imdb_latent_30pct_fixl_dev/explanations.txt ../spec/data/explanations-dev/imdb_latent_30pct_fixl_dev.txt
cp explanations/imdb_bernoulli_sparsity0003_fixl_dev/explanations.txt ../spec/data/explanations-dev/imdb_bernoulli_sparsity0003_fixl_dev.txt
cp explanations/snli_latent_10pct_fixl_dev/explanations.txt ../spec/data/explanations-dev/snli_latent_10pct_fixl_dev.txt
cp explanations/snli_bernoulli_sparsity0003_fixl_dev/explanations.txt ../spec/data/explanations-dev/snli_bernoulli_sparsity0003_fixl_dev.txt


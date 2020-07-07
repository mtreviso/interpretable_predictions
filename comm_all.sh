#echo "SST"
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.sst.communicate --ckpt results/sst/latent_30pct --save_path results/comm_sst/latent_30pct
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.sst.communicate --ckpt results/sst/bernoulli_sparsity01 --save_path results/comm_sst/bernoulli_sparsity01
#
#echo "IMDB"
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.imdb.communicate --ckpt results/imdb/latent_30pct --save_path results/comm_imdb/latent_30pct
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.imdb.communicate --ckpt results/imdb/bernoulli_sparsity0003 --save_path results/comm_imdb/bernoulli_sparsity0003
#
#echo "AGNEWS"
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.agnews.communicate --ckpt results/agnews/latent_30pct --save_path results/comm_agnews/latent_30pct
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.agnews.communicate --ckpt results/agnews/bernoulli_sparsity01 --save_path results/comm_agnews/bernoulli_sparsity01

#echo "SNLI"
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.snli.communicate --ckpt results/snli/latent_10pct --save_path results/comm_snli/latent_10pct
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.snli.communicate --ckpt results/snli/da --save_path results/comm_snli/da


#echo "YELP"
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.yelp.communicate --ckpt results/yelp/latent_30pct --save_path results/comm_yelp/latent_30pct
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.yelp.communicate --ckpt results/yelp/bernoulli_sparsity0003 --save_path results/comm_yelp/bernoulli_sparsity0003
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.yelp.communicate --ckpt results/yelp/bernoulli_sparsity01 --save_path results/comm_yelp/bernoulli_sparsity01


# new snli
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.snli.communicate --ckpt results/snli/bernoulli_sparsity01_fix --save_path results/comm_snli/bernoulli_sparsity01_fix --epochs 0


# ACC: 0.9106   | dev: 0.9184
#dev acc 0.9536
#dev true_acc 0.9120
#dev avg k:  24.96
#dev H: 11.38
#test acc 0.9472
#test true_acc 0.9016
#test avg k: 24.18
#human avg k: 24.33
#CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.communicate --ckpt results/imdb/latent_10pct --save_path results/comm_imdb/latent_10pct

# ACC: 0.8699  | dev: 0.8764
#dev acc 0.9188
#dev true_acc 0.8736
#dev avg k: 40.36
#dev h: 10.66
#test acc 0.9166
#test true_acc 0.8613
#human avg k: 39.40
#CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.communicate --ckpt results/imdb/bernoulli_sparsity001 --save_path results/comm_imdb/bernoulli_sparsity001


# testing (diverges)
#CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.communicate --ckpt results/imdb/bernoulli_sparsity003 --save_path results/comm_imdb/bernoulli_sparsity003




CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.sst.communicate --ckpt results/sst/bernoulli_sparsity01 --save_path results/tmp
CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.sst.communicate --ckpt results/sst/latent_30pct --save_path results/tmp

CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.imdb.communicate --ckpt results/imdb/bernoulli_sparsity001 --save_path results/tmp
CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.imdb.communicate --ckpt results/imdb/latent_10pct --save_path results/tmp

CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.agnews.communicate --ckpt results/agnews/bernoulli_sparsity01 --save_path results/tmp
CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.agnews.communicate --ckpt results/agnews/latent_30pct --save_path results/tmp

CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.yelp.communicate --ckpt results/yelp/bernoulli_sparsity01 --save_path results/tmp
CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.yelp.communicate --ckpt results/yelp/latent_30pct --save_path results/tmp

CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.snli.communicate --ckpt results/snli/bernoulli_sparsity0003_fix_new_indep_qk --save_path results/tmp
CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.snli.communicate --ckpt results/snli/latent_10pct --save_path results/tmp

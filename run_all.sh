#echo "SST"
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.sst.train --model latent --selection 0.3 --save_path results/sst/latent_30pct
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.sst.train --model rl --sparsity 0.01 --save_path results/sst/bernoulli_sparsity01

#echo "IMDB"
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.imdb.train --model latent --selection 0.3 --save_path results/imdb/latent_30pct
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.imdb.train --model rl --sparsity 0.0003 --save_path results/imdb/bernoulli_sparsity0003

#python3 -m latent_rationale.imdb.train --model rl --sparsity 0.0003 --coherence 2.0 --save_path results/imdb/bernoulli_sparsity01 --batch_size 64 \
#--weight_decay 0.000002 --dropout 0.2 --lr 0.0004 --lagrange_alpha 0.5 --lagrange_lr 0.05 --lambda_init 0.00001


#echo "AGNEWS"
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.agnews.train --model latent --selection 0.3 --save_path results/agnews/latent_30pct
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.agnews.train --model rl --sparsity 0.01 --save_path results/agnews/bernoulli_sparsity01

#echo "YELP"
# todo: if 0.01 doesnt work try 0.003
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.yelp.train --model rl --sparsity 0.003 --batch_size 256 --lr 0.001 --num_iterations -10 --save_path results/yelp/bernoulli_sparsity003

#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.yelp.train --model rl --sparsity 0.01 --batch_size 256 --lr 0.001 --num_iterations -10 --save_path results/yelp/bernoulli_sparsity01
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.yelp.train --model latent --selection 0.3 --batch_size 256 --lr 0.001 --num_iterations -10 --save_path results/yelp/latent_30pct

#echo "SNLI"
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.snli.train --model decomposable --dist hardkuma --selection 0.10 --save_path results/snli/latent_10pct
#CUDA_VISIBLE_DEVICES=3 python3 -m latent_rationale.snli.train --model decomposable --save_path results/snli/da


# running:
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.snli.train --model decomposable --dist bernoulli --sparsity 0.1 --save_path results/snli/bernoulli_sparsity1
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.snli.train --model decomposable --dist bernoulli --sparsity 0.0003 --save_path results/snli/bernoulli_sparsity0003

# fixed:
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.snli.train --model decomposable --dist bernoulli --sparsity 0.01 --save_path results/snli/bernoulli_sparsity01_fix
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.snli.train --model decomposable --dist bernoulli --sparsity 0.003 --save_path results/snli/bernoulli_sparsity003_fix
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.snli.train --model decomposable --dist bernoulli --sparsity 0.1 --save_path results/snli/bernoulli_sparsity1_fix



#CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.train --model latent --selection 0.10 --save_path results/imdb/latent_10pct
#CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.train --model latent --selection 0.05 --save_path results/imdb/latent_05pct
#CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.train --model rl --sparsity 0.01 --save_path results/imdb/bernoulli_sparsity01
#CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.train --model rl --sparsity 0.001 --save_path results/imdb/bernoulli_sparsity001
#CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.train --model rl --sparsity 0.1 --save_path results/imdb/bernoulli_sparsity1
#CUDA_VISIBLE_DEVICES=0 python3 -m latent_rationale.imdb.train --model rl --sparsity 0.003 --save_path results/imdb/bernoulli_sparsity003



# check run stats
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.sst.train --model rl --sparsity 0.01 --save_path results/tmp
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.sst.train --model latent --selection 0.3 --save_path results/tmp
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.snli.train --model decomposable --dist hardkuma --selection 0.10 --save_path results/tmp
#CUDA_VISIBLE_DEVICES=2 python3 -m latent_rationale.snli.train --model decomposable --dist bernoulli --sparsity 0.0003 --save_path results/tmp

python -m learning.gp.evaluate_models --n-gp-samples 500 --hdim 16 --bb-fname 100_eval_doors.pickle --type ngpucb20_factored_doors --Ls 20 20 20 --eval-method T --gp-fname gp_models/gp_20L_100M_factored.pt --nn-fname pretrained_models/doors/model_20L_100M.pt --N 10 --T 10


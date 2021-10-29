# nthai, 

# predict
cd /Users/hainguyen/Documents/nthai/ctu/nthai-2020/old-archived/PhD/workspace/deepmg_app/deepmg_v34
python __main__.py -i colphy -t fills  -z 255 --colormap gray --channel 1 --type_bin spb -a predict --model pretrained --pretrained_w_path  /Users/hainguyen/Documents/nthai/ctu/nthai-2020/old-archived/PhD/workspace/deepmg_app/results/colphy_fill_pix_r0p1.0spbm0.0a1.0rainbowbi10_0.0_1.0/models/a1_k2_nonecaffe_estopc5_fc_o1adam_lr-1.0de0.0e100_20210507_071141c255.0di-1ch3dfc0.0model_s1k1

# good parameters for minisom
python3 $file_run -i $db  -t minisom --alpha_v 0.5 --lr_visual 0.5 --iter_visual 100 --fig_size 48 --colormap gray --channel 1 --type_bin spb --model $model_v --parent_folder_results $res_fold



# some remarks:
# for reduction with PCA
n_components must be between 0 and min(n_samples, n_features)=47 

# some case may cause errors:
# k = 0,1


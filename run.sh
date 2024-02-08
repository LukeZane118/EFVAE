#!/bin/bash
dns=(lastfm citeulike-a filmtrust steam)
lrs=(5.e-3 0.01 5.e-4 5.e-3)
ncs=(64 64 16 32)
nss=(800 800 100 200)
for sn in Uni Pop FIMS-Uni FIMS-Pop
do
    for mn in ef-multi_vae ef-multi_dae ef-recvae
    do
        for i in 0 1 2 3
        do
            dn=${dns[i]}
            lr=${lrs[i]}
            nc=${ncs[i]}
            ns=${nss[i]}

            python ${mn}/main.py -dn ${dn} -ln [sampler_${sn}][num_sample_${ns}][num_cluster_${nc}][lr_${lr}][test] --lr ${lr} -sn ${sn} -ns ${ns} -nc ${nc} --epochs 200
        done
    done
done
#!/bin/bash

classifiers=(lr gbc tree gnb rf qda NN_1 NN_2 NN_3 NN_4 NN_5)
recourse_methods=(brute_force) # wachter growing_spheres genetic_search)

echo 'starting the Gaussian'
# Remove previous risks
rm checkpoints/gaussians_cluster/data/risk_before_after.csv

# Run tasks in parallel
for i in "${classifiers[@]}"; do
    for j in "${recourse_methods[@]}"; do
        python -u synthetic_experiments.py \
            --data_set gaussians \
            --classifier "$i" \
            --recourse "$j" &
    done
done 
wait

echo 'starting the Cirles'
# Remove previous risks
rm checkpoints/circles_cluster/data/risk_before_after.csv

# Run tasks in parallel
for i in "${classifiers[@]}"; do
    for j in "${recourse_methods[@]}"; do
        python -u synthetic_experiments.py \
            --data_set circles \
            --classifier "$i" \
            --recourse "$j" &
    done
done 
wait

echo 'starting the Moons'
# Remove previous risks
rm checkpoints/moons_cluster/data/risk_before_after.csv


# Run tasks in parallel
for i in "${classifiers[@]}"; do
    for j in "${recourse_methods[@]}"; do
        python -u synthetic_experiments.py \
            --data_set moons \
            --classifier "$i" \
            --recourse "$j" &
    done
done 
wait

echo 'finished'

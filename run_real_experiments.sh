#!/bin/bash

classifiers=(lr gbc tree gnb rf NN_1 NN_2 NN_3 NN_4 NN_5)
recourse_methods=(wachter growing_spheres genetic_search)

echo 'starting the Credit'
# Remove previous risks
rm checkpoints/credit_cluster/data/risk_before_after.csv

# Run tasks in parallel
for i in "${classifiers[@]}"; do
    for j in "${recourse_methods[@]}"; do
        python -u real_experiments.py \
            --data_set credit \
            --classifier "$i" \
            --recourse "$j" &
    done
done 
wait

echo 'starting the Adult'
# Remove previous risks
rm checkpoints/adult_cluster/data/risk_before_after.csv

# Run tasks in parallel
for i in "${classifiers[@]}"; do
    for j in "${recourse_methods[@]}"; do
        python -u real_experiments.py \
            --data_set adult \
            --classifier "$i" \
            --recourse "$j" &
    done
done 
wait

echo 'starting the Heloc'
# Remove previous risks
rm checkpoints/heloc_cluster/data/risk_before_after.csv

# Run tasks in parallel
for i in "${classifiers[@]}"; do
    for j in "${recourse_methods[@]}"; do
        python -u real_experiments.py \
            --data_set heloc \
            --classifier "$i" \
            --recourse "$j" &
    done
done 
wait

echo 'finished'

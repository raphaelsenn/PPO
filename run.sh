#!/bin/bash

for ((i=0;i<3;i+=1))
do
    python3 main.py \
        --env_id="LunarLander-v3" \
        --seed=$i \
        --num_timesteps=3000000 \
    
    python3 main.py \
        --env_id="BipedalWalker-v3" \
        --seed=$i \
        --num_timesteps=3000000 \

    python3 main.py \
        --env_id="CarRacing-v3" \
        --seed=$i \
        --num_timesteps=5000000 \

done
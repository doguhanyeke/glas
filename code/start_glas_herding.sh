#!/bin/bash

python collision_avoidance_node.py --robot_id 1 --initial_x 3.0 --initial_y 6.0 --instance ./config/herding_glas.yaml &
python collision_avoidance_node.py --robot_id 2 --initial_x 3.0 --initial_y 4.0 --instance ./config/herding_glas.yaml &
python collision_avoidance_node.py --robot_id 3 --initial_x 3.0 --initial_y 2.0 --instance ./config/herding_glas.yaml &

# terminator -T "<node-1>" -e "cd ../..; . install/setup.bash; cd ./src/glas/code; python collision_avoidance_node.py --robot_id 1" &
# sleep 1

# terminator -T "<node-2>" -e "cd ../..; . install/setup.bash; cd ./src/glas/code; python collision_avoidance_node.py --robot_id 2" &
# sleep 1

# terminator -T "<node-3>" -e "cd ../..; . install/setup.bash; cd ./src/glas/code; python collision_avoidance_node.py --robot_id 3" &
# sleep 1

# bash

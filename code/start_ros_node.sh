#!/bin/bash

python collision_avoidance_node.py --robot_id 1 &
python collision_avoidance_node.py --robot_id 2 &
python collision_avoidance_node.py --robot_id 3 &

# terminator -T "<node-1>" -e "cd ../..; . install/setup.bash; cd ./src/glas/code; python collision_avoidance_node.py --robot_id 1" &
# sleep 1

# terminator -T "<node-2>" -e "cd ../..; . install/setup.bash; cd ./src/glas/code; python collision_avoidance_node.py --robot_id 2" &
# sleep 1

# terminator -T "<node-3>" -e "cd ../..; . install/setup.bash; cd ./src/glas/code; python collision_avoidance_node.py --robot_id 3" &
# sleep 1

# bash

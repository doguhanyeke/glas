#!/bin/bash
python collision_avoidance_node.py --robot_id 1 --initial_x 3.0 --initial_y 6.0 --instance ./config/navdelay_glas.yaml &
python collision_avoidance_node.py --robot_id 2 --initial_x 3.0 --initial_y 4.0 --instance ./config/navdelay_glas.yaml &
python collision_avoidance_node.py --robot_id 3 --initial_x 3.0 --initial_y 2.0 --instance ./config/navdelay_glas.yaml &
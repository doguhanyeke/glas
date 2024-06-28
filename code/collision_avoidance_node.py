#!/usr/bin/env python

__author__ = "Kartik Anand Pant"
__contact__ = "kpant14@gmail.com"

import os
import torch
import rclpy
import numpy as np
from examples.run import parse_args
from examples.run_singleintegrator import SingleIntegratorParam
from examples import run_singleintegrator
from systems.singleintegrator import SingleIntegrator
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleLocalPosition
from functools import partial


class CollisionAvoidanceSystem(Node):
    def __init__(self):
        # initiate glas
        super().__init__("ros2_glas")
        # set publisher and subscriber quality of service profile
        qos_profile_pub = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            durability = QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth = 1
        )
        qos_profile_sub = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            durability = QoSDurabilityPolicy.VOLATILE,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth = 1
        )
        # self.declare_parameter('robot_id', 1)
        # self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        # self.robot_id =1
        # parse the config file
        # variables = self.parse_config(
        #     "/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/config.txt")
        # self.numRobots = int(variables['num_robots'])
        # self.timeStep = float(variables['time_step'])
        # self.environmentFile = variables['environment_file']
        # self.radius = float(variables['radius'])

        self.args = parse_args()

        self.robot_id = self.args.robot_id
        self.initial_x = self.args.initial_x
        self.initial_y = self.args.initial_y
        print(self.robot_id, self.initial_x, self.initial_y)
        # print("here", self.args)
        self.param = SingleIntegratorParam()
        # print("here", self.param)
        # print("heyo", self.param.n_agents)

        self.env = SingleIntegrator(self.param)
        # print("here", self.env)

        print(self.param.il_train_model_fn)
        self.controllers = {
            'current': torch.load(self.param.il_train_model_fn), }
        # print("here", self.controllers)
        self.param.default_instance = self.args.instance
        self.s0 = run_singleintegrator.load_instance(
            self.param, self.env, self.args.instance)

        # print("heyo3", self.param.n_agents)

        self.observations = []
        self.reward = 0

        self.states = np.empty((len(self.param.sim_times), self.env.n))
        self.actions = np.empty((len(self.param.sim_times)-1, self.env.m))

        self.env.reset(self.s0)
        self.states[0] = np.copy(self.env.s)

        self.done = False
        self.step = 0

        self.robot_initial_positions = []
        self.goal_positions = []
        self.obstacles = []

        torch.set_num_threads(1)
        if self.s0 is None:
            self.s0 = self.env.reset()

        self.sim_results = []

        # print("heyo22", self.env.agents)
        # print(self.env.agents)

        self.vehicle_local_pos_subs = [{'local_pos_sub':None} for _ in range(len(self.env.agents))]
        for i in range(len(self.env.agents)):
            self.vehicle_local_pos_subs[i]['local_pos_sub'] = self.create_subscription(
                VehicleLocalPosition,
                'px4_' + str(i+1) + '/fmu/out/vehicle_local_position',
                partial(self.vehicle_local_pos_callback,id=i),                             # instead of lambda function lambda msg: self.vehicle_status_callback(msg,id=i), use partial function
                qos_profile_sub)
        self.trajectory_setpoint_pub_ = self.create_publisher(
            TrajectorySetpoint, 
            'px4_'+ str(self.robot_id) + '/fmu/orca2/trajectory_setpoint', 
            qos_profile_pub)
        
    def vehicle_local_pos_callback(self, msg, id):
        # TODO: not assign, add
        
        self.set_agent_position(id, msg.y + self.initial_x, msg.x + self.initial_y)

        self.calculate_next_velocities()
        trajectory_msg = TrajectorySetpoint()

        vx = self.get_agent_velocities()[self.robot_id-1][0]
        vy = self.get_agent_velocities()[self.robot_id-1][1]
        px = self.get_agent_positions()[self.robot_id-1][0]
        py = self.get_agent_positions()[self.robot_id-1][1]
        print(self.robot_id, px, py, vx, vy)
        trajectory_msg.velocity[1]  = vx
        trajectory_msg.velocity[0]  = vy
        trajectory_msg.velocity[2]  = 0
        
        self.trajectory_setpoint_pub_.publish(trajectory_msg)

    def parse_config(self, file_path):
        variables = {}
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    key_value = line.split('=')
                    if len(key_value) == 2:
                        key, value = key_value
                        key = key.strip()
                        value = value.strip().strip('"')
                        variables[key] = value
        return variables

    def reached_goal(self):
        # check if the agents have reached the goal
        return self.env.done()

    def set_agent_position(self, agent_id, x, y):
        self.env.update_agent_pos(agent_id, x, y)

    # TODO: set_vel may not be working
    def set_agent_velocity(self, agent_id, vx, vy):
        self.env.update_agent_vel(agent_id, vx, vy)

    def get_agent_positions(self):
        # get the agent positions
         # get the agent velocities
        positions = []
        for agent in self.env.agents:
            positions.append(agent.p)
        return positions
       

    def get_agent_velocities(self):
        # get the agent velocities
        velocities = []
        for agent in self.env.agents:
            velocities.append(agent.v)
        return velocities

    def print_agent_positions(self):
        # print the agent positions
        print("Agent positions")
        for agent_position in self.get_agent_positions():
            print(agent_position)

    def print_agent_velocities(self):
        print("Agent velocities")
        for agent_velocity in self.get_agent_velocities():
            print(agent_velocity)

    # def calculate_next_velocities_(self, agentID, agentPos):
    #     state = self.states[self.step]
    #     observation = self.env.observe()

    #     for name, controller in self.controllers.items():
    #         action = controller.policy(observation)
    #         next_state, r, done, _ = self.env.step_(
    #             action, False, agentID, agentPos)
    #         self.reward += r

    #         self.states[self.step + 1] = next_state
    #         self.actions[self.step] = action.flatten()
    #         self.observations.append(observation)

    #         self.step += 1
    #         self.done = done
    #     return self.states, self.observations, self.actions, self.step

    def calculate_next_velocities(self,):
        # calculate the next velocities of the agents
        state = self.states[self.step]
        observation = self.env.observe()

        for name, controller in self.controllers.items():
            action = controller.policy(observation)
            next_state, r, done, _ = self.env.step(
                action, compute_reward=False)
            self.reward += r

            self.states[self.step + 1] = next_state
            self.actions[self.step] = action.flatten()
            self.observations.append(observation)

            self.step += 1
            self.done = done
        return self.states, self.observations, self.actions, self.step
    
    def write_positions_to_file(self, time_step):
        # write the agent positions to a file
        with open('agent_positions.txt', 'a') as f:
            f.write(str(time_step) + " ")
            agent_pos_list = self.get_agent_positions()
            for i in range(0, len(agent_pos_list), 2):
                # write with .2f precision
                f.write("({:.4f},{:.4f}) ".format(
                    agent_pos_list[i], agent_pos_list[i+1]))

            f.write("\n")

    def delete_file(self, ):
        # delete the file if exists
        if os.path.exists("agent_positions.txt"):
            os.remove("agent_positions.txt")

def main():
    rclpy.init(args=None)
    ros2_glas = CollisionAvoidanceSystem()
    rclpy.spin(ros2_glas)
    ros2_glas.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()




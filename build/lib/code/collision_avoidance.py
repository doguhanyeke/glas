from examples import run_singleintegrator
from systems.singleintegrator import SingleIntegrator
from examples.run import run, parse_args
import torch
import numpy as np
from collections import namedtuple
from sim import run_sim
import yaml
import plotter
from matplotlib.patches import Rectangle, Circle

from examples.run_singleintegrator import SingleIntegratorParam

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from geometry_msgs.msg import PoseStamped


class CollisionAvoidanceSystem:
    def __init__(self):
        # initiate glas

        # parse the config file
        variables = self.parse_config(
            "/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/config.txt")
        self.numRobots = int(variables['num_robots'])
        self.timeStep = float(variables['time_step'])
        self.environmentFile = variables['environment_file']
        self.radius = float(variables['radius'])

        self.args = parse_args()
        # print("here", self.args)
        self.param = SingleIntegratorParam()
        # print("here", self.param)
        self.env = SingleIntegrator(self.param)
        # print("here", self.env)

        # print(self.param.il_train_model_fn)
        self.controllers = {
            'current': torch.load(self.param.il_train_model_fn), }
        # print("here", self.controllers)

        self.s0 = run_singleintegrator.load_instance(
            self.param, self.env, self.args.instance)

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

    def set_initial_positions(self, file_path):
        # set the initial positions of the agents
        self.robot_initial_positions = self.read_coordinates_from_file(
            file_path=file_path)

    def read_coordinates_from_file(self, file_path):
        coordinates = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(', ')
                coordinates.append((int(parts[0]), int(parts[1])))
        return coordinates

    def read_float_coordinates_from_file(file_path):
        coordinates = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(', ')
                coordinates.append((float(parts[0]), float(parts[1])))
        return coordinates

    def set_goal_positions(self, file_path):
        self.goal_positions = self.read_float_coordinates_from_file(
            file_path=file_path)

    def set_obstacle_positions(self, yaml_file_path, new_obstacles):
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Update the obstacles part
        data['map']['obstacles'] = new_obstacles

        with open(yaml_file_path, 'w') as file:
            yaml.safe_dump(data, file)

    def reached_goal(self):
        # check if the agents have reached the goal
        return self.done

    def set_agent_positions(self, agent_positions):
        # set the agent positions
        self.sim_results[-1].states = agent_positions

    def get_agent_positions(self):
        # get the agent positions
        if len(self.sim_results) != 0:
            return self.sim_results[-1].states[self.step]

    def get_agent_velocities(self):
        # get the agent velocities
        velocities = []
        for agent in self.env.agents:
            print(agent.i)
            print(len(self.sim_results[-1].actions[0]))
            print(self.sim_results[-1].actions[0])
            print(self.sim_results[-1].actions[1])
            print(self.sim_results[-1].actions[2])
            print(self.sim_results[-1].actions[3])
            print(len(self.sim_results[-1].actions))
            for v in self.sim_results[-1].actions[:5]:
                print(v)
            # print(self.sim_results[self.step].actions[0])
            print(self.sim_results[-1].actions[0][0])
            print(self.sim_results[-1].actions[1])
            # print(len(self.sim_results[-1].actions[0]))
            velocities.append(
                (self.sim_results[self.step - 1].actions[self.step, 2*agent.i+0],
                 self.sim_results[self.step - 1].actions[self.step, 2*agent.i+1]))
        return velocities

    def set_agent_velocities(self, agent_velocities):
        # set the agent velocities
        for ind, agent in enumerate(self.env.agents):
            self.sim_results[-1].actions[self.step, 2 *
                                         agent.i + 0] = agent_velocities[ind][0]
            self.sim_results[-1].actions[self.step, 2 *
                                         agent.i + 1] = agent_velocities[ind][1]

    def print_agent_positions(self):
        # print the agent positions
        print("Agent positions")
        if self.get_agent_positions() is None:
            print("No agent positions found")
            return
        for agent_position in self.get_agent_positions():
            print(agent_position)

    def print_agent_velocities(self):
        # print the agent velocities
        print("Agent velocities")
        if self.get_agent_velocities() is None:
            print("No agent velocities found")
            return
        for agent_velocity in self.get_agent_velocities():
            print(agent_velocity)

    def calculate_next_velocities(self, agentID, agentPos):
        # calculate the next velocities of the agents
        self.sim_results[-1].states[self.step][self.env.agent_idx_to_state_idx(
            agentID)] = agentPos

        state = self.states[self.step]
        observation = self.env.observe()

        for name, controller in self.controllers.items():
            action = controller.policy(observation)
            next_state, r, done, _ = self.env.step(
                action, compute_reward=False)
            reward += r

            self.states[self.step + 1] = next_state
            self.actions[self.step] = action.flatten()
            self.observations.append(observation)

            self.step += 1
            self.done = done

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

    def pack_results(self):
        # states = np.empty((len(self.param.sim_times), self.env.n))
        # states[0] = np.copy(self.env.s)
        # actions = np.empty((len(self.param.sim_times)-1, self.env.m))
        # observations = []
        # reward = 0
        # controller_name = None
        # for name, controller in self.controllers.items():
        #     for i in range(len(self.sim_results)):
        #         states[i+1] = self.sim_results[i].states
        #         actions[i] = self.sim_results[i].actions
        #         observations.append(self.sim_results[i].observations)
        #         reward += self.sim_results[i].reward
        #         controller_name = name

        # return SimResult._make(self.states, self.observations, self.actions, self.step + (controller_name, ))
        big_list = []
        for result in self.sim_results:
            big_list.append(result)

    def draw(self):
        # plot state space
        times = self.param.sim_times
        result = self.pack_results()
        if self.param.env_name in ['SingleIntegrator', 'SingleIntegratorVelSensing', 'DoubleIntegrator']:
            fig, ax = plotter.make_fig()
            ax.set_title('State Space')
            ax.set_aspect('equal')

            for o in self.env.obstacles:
                ax.add_patch(
                    Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))

            for agent in self.env.agents:

                line = ax.plot(result.states[0:result.steps, self.env.agent_idx_to_state_idx(agent.i)],
                               result.states[0:result.steps, self.env.agent_idx_to_state_idx(agent.i)+1], alpha=0.5)
                color = line[0].get_color()

                # plot velocity vectors:
                X = []
                Y = []
                U = []
                V = []
                for k in np.arange(0, result.steps, 100):
                    X.append(
                        result.states[k, self.env.agent_idx_to_state_idx(agent.i)])
                    Y.append(
                        result.states[k, self.env.agent_idx_to_state_idx(agent.i)+1])
                    if self.param.env_name in ['SingleIntegrator', 'SingleIntegratorVelSensing']:
                        # Singleintegrator: plot actions
                        U.append(result.actions[k, 2*agent.i+0])
                        V.append(result.actions[k, 2*agent.i+1])
                    elif self.param.env_name in ['DoubleIntegrator']:
                        # doubleintegrator: plot velocities
                        U.append(
                            result.states[k, self.env.agent_idx_to_state_idx(agent.i)+2])
                        V.append(
                            result.states[k, self.env.agent_idx_to_state_idx(agent.i)+3])

                ax.quiver(X, Y, U, V, angles='xy', scale_units='xy',
                          scale=0.5, color=color, width=0.005)
                plotter.plot_circle(result.states[1, self.env.agent_idx_to_state_idx(agent.i)],
                                    result.states[1, self.env.agent_idx_to_state_idx(agent.i)+1], self.param.r_agent, fig=fig, ax=ax, color=color)
                plotter.plot_square(
                    agent.s_g[0], agent.s_g[1], self.param.r_agent, angle=45, fig=fig, ax=ax, color=color)

            # draw state for each time step
            robot = 0
            if self.param.env_name in ['SingleIntegrator']:
                for step in np.arange(0, result.steps, 1000):
                    fig, ax = plotter.make_fig()
                    ax.set_title('State at t={} for robot={}'.format(
                        times[step], robot))
                    ax.set_aspect('equal')

                    # plot all obstacles
                    for o in self.env.obstacles:
                        ax.add_patch(
                            Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))

                    # plot overall trajectory
                    line = ax.plot(result.states[0:result.steps, self.env.agent_idx_to_state_idx(robot)],
                                   result.states[0:result.steps, self.env.agent_idx_to_state_idx(robot)+1], "--")
                    color = line[0].get_color()

                    # plot current position
                    plotter.plot_circle(result.states[step, self.env.agent_idx_to_state_idx(robot)],
                                        result.states[step, self.env.agent_idx_to_state_idx(robot)+1], self.param.r_agent, fig=fig, ax=ax, color=color)

                    # plot current observation
                    observation = result.observations[step][robot][0]
                    num_neighbors = int(observation[0])
                    num_obstacles = int(
                        (observation.shape[0]-3 - 2*num_neighbors)/2)

                    robot_pos = result.states[step, self.env.agent_idx_to_state_idx(
                        robot):self.env.agent_idx_to_state_idx(robot)+2]

                    idx = 3
                    for i in range(num_neighbors):
                        pos = observation[idx: idx+2] + robot_pos
                        ax.add_patch(
                            Circle(pos, 0.25, facecolor='gray', edgecolor='red', alpha=0.5))
                        idx += 2

                    for i in range(num_obstacles):
                        # pos = observation[idx : idx+2] + robot_pos - np.array([0.5,0.5])
                        # ax.add_patch(Rectangle(pos, 1.0, 1.0, facecolor='gray', edgecolor='red', alpha=0.5))
                        pos = observation[idx: idx+2] + robot_pos
                        ax.add_patch(
                            Circle(pos, 0.5, facecolor='gray', edgecolor='red', alpha=0.5))
                        idx += 2

                    # plot goal
                    goal = observation[1:3] + robot_pos
                    ax.add_patch(
                        Rectangle(goal - np.array([0.2, 0.2]), 0.4, 0.4, alpha=0.5, color=color))

                # 	# import matplotlib.pyplot as plt
                # 	# plt.savefig("test.svg")
                # 	# exit()

        # plot time varying states
        if self.param.env_name in ['SingleIntegrator', 'DoubleIntegrator']:
            for i_config in range(self.env.state_dim_per_agent):
                fig, ax = plotter.make_fig()
                ax.set_title(self.env.states_name[i_config])
                for agent in self.env.agents:
                    for result in self.sim_results:
                        ax.plot(
                            times[1:result.steps],
                            result.states[1:result.steps, self.env.agent_idx_to_state_idx(
                                agent.i)+i_config],
                            label=result.name)

        # plot time varying actions
        if self.param.env_name in ['SingleIntegrator', 'DoubleIntegrator']:
            for i_config in range(self.env.action_dim_per_agent):
                fig, ax = plotter.make_fig()
                ax.set_title(self.env.actions_name[i_config])
                for agent in self.env.agents:
                    for result in self.sim_results:
                        ax.plot(
                            times[1:result.steps],
                            result.actions[1:result.steps, agent.i *
                                           self.env.action_dim_per_agent+i_config],
                            label=result.name)

                        #
                        if i_config == 5:
                            ax.set_yscale('log')

        plotter.save_figs(self.param.plots_fn)
        plotter.open_figs(self.param.plots_fn)

        # visualize
        self.env.visualize(self.sim_results[0].states[0:result.steps], 0.1)


def main(args=None):
    c = CollisionAvoidanceSystem()
    variables = c.parse_config(
        "/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/config.txt")
    print(variables)

    # while (c.reached_goal() is False):
    for i in range(5):
        SimResult = namedtuple(
            'SimResult', ['states', 'observations', 'actions', 'steps'])
        result = SimResult._make(c.calculate_next_velocities())
        c.sim_results.append(result)
        c.print_agent_positions()
        # c.print_agent_velocities()
    c.draw()

if __name__ == '__main__':
    main()

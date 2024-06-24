import logging
import math
import os
from typing import Any, List
import numpy as np
# import plotly.graph_objects as go
from staliro.core.interval import Interval
from staliro.core import best_eval, best_run
from staliro.core.model import FailureResult
from staliro.core.result import worst_eval, worst_run
from staliro.models import State, ode, blackbox, BasicResult, Model, ModelInputs, ModelResult, Trace
from staliro.optimizers import UniformRandom, DualAnnealing
from staliro.options import Options
from staliro.specifications import RTAMTDense, RTAMTDiscrete
from staliro.staliro import simulate_model, staliro
import math
# from collision_avoidance import CollisionAvoidanceSystem

import os
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
import pickle

# Custom unpickler to handle module paths


class CollisionAvoidanceSystem:
    def __init__(self):
        # initiate glas

        # parse the config file
        # variables = self.parse_config(
        #     "/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/config.txt")
        # self.numRobots = int(variables['num_robots'])
        # self.timeStep = float(variables['time_step'])
        # self.environmentFile = variables['environment_file']
        # self.radius = float(variables['radius'])

        self.args = parse_args()
        # print("here", self.args)
        self.param = SingleIntegratorParam()
        # print("here", self.param)
        # print("heyo", self.param.n_agents)

        self.env = SingleIntegrator(self.param)
        # print("here", self.env)

        # print(self.param.il_train_model_fn)
        print("here ", self.param.il_train_model_fn)
        self.controllers = {
            # Use custom load function
            'current': torch.load(self.param.il_train_model_fn),
        }

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
        return self.env.s

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

    def calculate_next_velocities_(self, agentID, agentPos):
        state = self.states[self.step]
        observation = self.env.observe()

        for name, controller in self.controllers.items():
            action = controller.policy(observation)
            next_state, r, done, _ = self.env.step_(
                action, False, agentID, agentPos)
            self.reward += r

            self.states[self.step + 1] = next_state
            self.actions[self.step] = action.flatten()
            self.observations.append(observation)

            self.step += 1
            self.done = done
        return self.states, self.observations, self.actions, self.step

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


# system parameters
time_step = 0.5
obstacles = []
goal_positions = []
radius = 0.15
num_of_robots = 3
# attack parameters
attackType = 3
delayConstant = 2
totalTimeStep = 0.0
victimRobotId = 1
attackedRobotId = 0
pointX = 21.0
pointY = 14.0
deadlockTimestep = 10
deadlockPosChange = 0.1
falsificationIterations = 1000
numberOfFalseMessage = 50
environment_name = "GlasEnvironment1"
falsificationRuns = 1
maxSpeed = 0.5


# change 5 if you want to change the number of parameters staliro wants to predict
def process_initial_conditions(initial_conditions):
    print("Initial conditions: ", initial_conditions)
    result = {}
    for i in range(0, len(initial_conditions), 3):
        key = round_to_nearest_quarter(initial_conditions[i])
        values = tuple(initial_conditions[i+1:i+3])
        result[key] = values
    return result


def load_obstacles_from_yaml(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)

    obstacles = data.get('map', {}).get('obstacles', [])
    return obstacles


def load_goals_from_yaml(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)

    goals = [agent.get('goal') for agent in data.get('agents', [])]
    return goals


def round_to_nearest_quarter(number):
    return round(number / time_step) * time_step


def calculate_abs_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def min_distance_square_circle(corner_point, circle_center):
    # Calculate the minimum distance between a square obstacle and a circular drone.
    square_x_min, square_y_min = corner_point
    square_x_max = square_x_min + 1 if square_x_min >= 0 else square_x_min - 1
    square_y_max = square_y_min + 1 if square_y_min >= 0 else square_y_min - 1

    circle_x, circle_y = circle_center

    # Calculate the closest point on the square to the circle center
    closest_x = np.clip(circle_x, min(
        square_x_min, square_x_max), max(square_x_min, square_x_max))
    closest_y = np.clip(circle_y, min(
        square_y_min, square_y_max), max(square_y_min, square_y_max))

    # Calculate the distance between the circle center and the closest point
    distance = np.sqrt((closest_x - circle_x) ** 2 +
                       (closest_y - circle_y) ** 2)

    # Subtract the circle's radius to get the distance from the edge of the circle
    # distance -= circle_radius

    # Ensure distance is non-negative (when the circle intersects the square)
    distance = max(0, distance)

    return distance


# create log of robot positions for last falsification
global_log_list_last = []
global_log_list_prev = []
global_log_list_fake = []
global_log_list_last_with_velocities = []

# create log of robot positions
global_log_list = []

global_min_dist_of_victim_to_static_so_far = float('inf')
global_min_dist_of_any_to_static_so_far = float('inf')
global_min_dist_of_victim_to_robots_so_far = float('inf')
global_min_dist_of_any_to_robots_so_far = float('inf')
global_min_navigating_to_point_victim = float('inf')
global_min_navigating_to_point_any = float('inf')
global_min_deadlock_of_victim_robot = float('inf')
global_min_deadlock_of_any_robot = float('inf')
global_min_navigation_duration_of_victim_robot = float('inf')
global_max_navigation_duration_of_any_robot = -float('inf')


@blackbox()
def Glas_Model(state: State, time: Interval, _: Any) -> BasicResult:
    global totalTimeStep
    global global_log_list_last
    global global_log_list_prev
    global global_log_list_fake
    global global_log_list_last_with_velocities

    global global_log_list
    global global_min_dist_of_victim_to_static_so_far
    global global_min_dist_of_any_to_static_so_far
    global global_min_dist_of_victim_to_robots_so_far
    global global_min_dist_of_any_to_robots_so_far
    global global_min_navigating_to_point_victim
    global global_min_navigating_to_point_any
    global global_min_deadlock_of_victim_robot
    global global_min_deadlock_of_any_robot
    global global_min_navigation_duration_of_victim_robot
    global global_max_navigation_duration_of_any_robot

    counter = 0.0

    time_result = []
    list_min_obstacle_dist_of_victim = []
    list_min_obstacle_dist_of_any_robot = []
    list_min_inter_robot_dist_of_victim_robot = []
    list_min_inter_robot_dist_of_any_robot = []
    list_min_deadlock_of_victim_robot = []
    list_min_deadlock_of_any_robot = []
    list_min_navigation_duration_of_victim_robot = []
    list_max_navigation_duration_of_any_robot = []
    list_min_navigating_to_point_victim = []
    list_min_navigating_to_point_any = []
    list_min_deadlock_for_navigation_duration_of_any_robot = []

    last_positions = []
    last_positions_dict = {robot_id: [] for robot_id in range(num_of_robots)}

    glas_instance = CollisionAvoidanceSystem()
    glas_instance.delete_file()
    totalTimeStep = glas_instance.param.sim_times[-1]

    mapped_conditions = process_initial_conditions(state)
    print("Mapped conditions: ")
    print(mapped_conditions)
    keys_list_times = list(mapped_conditions.keys())

    local_dist_victim = float('inf')
    local_pos_list_for_dist_victim = []
    local_dist_any = float('inf')
    local_pos_list_for_dist_any = []
    local_min_obstacle_dist_of_victim = float('inf')
    local_pos_list_for_obstacle_dist_of_victim = []
    local_min_obstacle_dist_of_any_robot = float('inf')
    local_pos_list_for_obstacle_dist_of_any_robot = []
    local_min_inter_robot_dist_of_victim_robot = float('inf')
    local_pos_list_for_inter_robot_dist_of_victim_robot = []
    local_min_inter_robot_dist_of_any_robot = float('inf')
    local_pos_list_for_inter_robot_dist_of_any_robot = []
    local_min_deadlock_of_victim_robot = float('inf')
    local_pos_list_for_deadlock_of_victim_robot = []
    local_min_deadlock_of_any_robot = float('inf')
    local_pos_list_for_deadlock_of_any_robot = []
    local_min_navigation_duration_of_victim_robot = float('inf')
    local_pos_list_for_navigation_duration_of_victim_robot = []
    local_max_navigation_duration_of_any_robot = -float('inf')
    local_pos_list_for_navigation_duration_of_any_robot = []

    delay_constant = 1
    # check if the attack is navigation delay attack
    if attackType == 9 or attackType == 10:
        delay_constant = delayConstant

    try:
        temp_list_for_global = []
        temp_list_for_global_prev = []
        temp_list_for_global_fake = []
        temp_list_for_global_with_velocities = []
        # run orca simulation with timestep
        ii = -1
        while counter < glas_instance.param.sim_times[-1]:
            ii += 1
            # while counter < (totalTimeStep * delay_constant):
            time_result.append(counter)
            # print("=======")
            # print("Counter: ", counter)

            updated_positions = np.copy(glas_instance.get_agent_positions())

            agent_pos = np.copy(glas_instance.get_agent_positions()[
                attackedRobotId * 2:attackedRobotId * 2 + 2])
            agent_original_pos = np.copy(glas_instance.get_agent_positions()[
                attackedRobotId * 2:attackedRobotId * 2 + 2])
            # print("Prev pos: ", agent_original_pos)

            # print real logs
            temp_list_for_global_prev.append(counter)
            for updated_position in updated_positions:
                temp_list_for_global_prev.append(updated_position)

            # plan attack
            print("Counter: ", counter)
            if counter in keys_list_times:

                # print("in attack plan")
                value = mapped_conditions[counter]
                print("Value: ", value)
                # set attacker commands
                agent_pos[0] += value[0]
                # updated_positions[attackedRobotId * 2 +
                #                   0] = updated_positions[attackedRobotId * 2 + 0] + value[0]
                agent_pos[1] += value[1]
                # updated_positions[attackedRobotId * 2 +
                #                   1] = updated_positions[attackedRobotId * 2 + 1] + value[1]
                updated_positions[attackedRobotId * 2] = agent_pos[0]
                updated_positions[attackedRobotId * 2 + 1] = agent_pos[1]
                # print("Updated positions1: ", updated_positions)
                glas_instance.set_agent_position(
                    attackedRobotId, agent_pos[0], agent_pos[1])

                # updated_velocities[attackedRobotId] = (
                # value[2], value[3])

            # print fake logs
            temp_list_for_global_fake.append(counter)
            for updated_position in updated_positions:
                temp_list_for_global_fake.append(updated_position)

            # glas_instance.calculate_next_velocities_(attackedRobotId, prev_pos)
            result = glas_instance.calculate_next_velocities_(
                attackedRobotId, agent_original_pos)
            SimResult = namedtuple(
                'SimResult', ['states', 'observations', 'actions', 'steps'])
            sim_result = SimResult._make(result)
            glas_instance.sim_results.append(sim_result)

            # print("Updated positions3: ", updated_positions)
            updated_positions = glas_instance.get_agent_positions()
            # print("Updated positions4: ", updated_positions)

            # ==========================================
            # Attack measurements
            # ==========================================
            # ==========================================
            # decide collision of victim robot with obstacles:
            # min_distance_to_obstacle = float('inf')
            # victim_position = updated_positions[victimRobotId *
            #                                     2:victimRobotId*2+2]
            # for robot_id, robot_pos_ind in enumerate(updated_positions[::2]):
            #     if robot_id == victimRobotId or robot_id == attackedRobotId:
            #         continue  # Skip the victim robot

            #     # Calculate the Euclidean distance
            #     robot_pos = [updated_positions[robot_id *
            #                                    2], updated_positions[robot_id*2+1]]
            #     distance = ((robot_pos[0] - victim_position[0]) ** 2 +
            #                 (robot_pos[1] - victim_position[1]) ** 2) ** 0.5
            #     # check all robots
            #     min_distance_to_obstacle = min(
            #         min_distance_to_obstacle, distance)

            #     # save for falsification
            #     list_min_obstacle_dist_of_victim.append(
            #         min_distance_to_obstacle)

            #     local_min_obstacle_dist_of_victim = min(
            #         local_min_obstacle_dist_of_victim, min_distance_to_obstacle)
            #     local_pos_list_for_obstacle_dist_of_victim.append(counter)
            #     for updated_position in updated_positions:
            #         local_pos_list_for_obstacle_dist_of_victim.append(
            #             updated_position)

            # ==========================================
            # V2: decide collision of victim robot with obstacles:
            print("R2O Collision Attack")
            min_distance_to_obstacle = float('inf')
            for obstacle in obstacles:
                victim_pos = [updated_positions[victimRobotId*2],
                              updated_positions[victimRobotId*2+1]]
                distance = min_distance_square_circle(
                    obstacle, victim_pos)
                # decide min dist by checking each obstacle
                min_distance_to_obstacle = min(
                    min_distance_to_obstacle, distance)
                # print for me
                # if min_distance_to_obstacle < radius:
                # print("Collision with static obstacle at time:", counter, " for robot",
                #   victimRobotId, "with position", updated_positions[victimRobotId])

            # save for falsification
            list_min_obstacle_dist_of_victim.append(min_distance_to_obstacle)

            local_min_obstacle_dist_of_victim = min(
                local_min_obstacle_dist_of_victim, min_distance_to_obstacle)
            local_pos_list_for_obstacle_dist_of_victim.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_obstacle_dist_of_victim.append(
                    updated_position)
            print("hey4")

            """
            # ==========================================
            # decide collision of any robot with obstacles:
            min_dist_of_any_robot_to_obstacles = float('inf')
            for robot_id, robot_pos in enumerate(updated_positions):
                if robot_id == attackedRobotId:
                    continue  # Skip the attacked robot

                for obstacle in obstacles:
                    for vertex in range(len(obstacle)):
                        # Each side of the obstacle is defined by two consecutive points
                        line_start = obstacle[vertex]
                        # Wrap around to the first point
                        line_end = obstacle[(vertex + 1) % len(obstacle)]

                        dist_to_obstacle = point_line_distance(
                            robot_pos, line_start, line_end)

                        # check each obstacle
                        min_dist_of_any_robot_to_obstacles = min(
                            min_dist_of_any_robot_to_obstacles, dist_to_obstacle)

                        # print for me
                        # if min_dist_of_any_robot_to_obstacles < radius:
                        # print("Collision with static obstacle at time:", counter,
                        #   " for robot", robot_id, "with position", robot_pos)
            # save for falsification
            list_min_obstacle_dist_of_any_robot.append(
                min_dist_of_any_robot_to_obstacles)

            local_min_obstacle_dist_of_any_robot = min(
                local_min_obstacle_dist_of_any_robot, min_dist_of_any_robot_to_obstacles)
            local_pos_list_for_obstacle_dist_of_any_robot.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_obstacle_dist_of_any_robot.append(
                    updated_position)
            """

            # ==========================================
            # decide collision of victim robot with other robots:
            print("R2R Collision Attack")
            min_dist_of_inter_robot_to_victim_robot = float('inf')
            victim_position = [updated_positions[victimRobotId*2],
                               updated_positions[victimRobotId*2+1]]
            for robot_id, robot_pos_ind in enumerate(updated_positions[::2]):
                if robot_id == victimRobotId or robot_id == attackedRobotId:
                    continue  # Skip the victim robot

                # Calculate the Euclidean distance
                robot_pos = [updated_positions[robot_id*2],
                             updated_positions[robot_id*2+1]]
                min_inter_robot_dist = ((robot_pos[0] - victim_position[0]) ** 2 +
                                        (robot_pos[1] - victim_position[1]) ** 2) ** 0.5
                # check all robots
                min_dist_of_inter_robot_to_victim_robot = min(
                    min_dist_of_inter_robot_to_victim_robot, min_inter_robot_dist)

                # print for me
                # if min_dist_of_inter_robot_to_victim_robot < radius * 2:
                # print("Collision with other robots at time:", counter, " between robots",
                #   victimRobotId, "and", robot_id, "with positions", victim_position, robot_pos)
            # save for falsification
            list_min_inter_robot_dist_of_victim_robot.append(
                min_dist_of_inter_robot_to_victim_robot)

            local_min_inter_robot_dist_of_victim_robot = min(
                local_min_inter_robot_dist_of_victim_robot, min_dist_of_inter_robot_to_victim_robot)
            local_pos_list_for_inter_robot_dist_of_victim_robot.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_inter_robot_dist_of_victim_robot.append(
                    updated_position)

            """
            # ==========================================
            # decide collision of any robot with other robots:
            min_dist_of_inter_robot_to_any_robot = float('inf')
            for robot_id, pos1 in enumerate(updated_positions):
                if robot_id == attackedRobotId:
                    continue  # Skip the attacked robot

                for j, pos2 in enumerate(updated_positions):
                    if j == robot_id or j == attackedRobotId:
                        continue  # Skip the same robot and the attacked robot

                    # Calculate the Euclidean distance
                    min_inter_robot_dist_of_any = ((pos1[0] - pos2[0]) ** 2 +
                                                   (pos1[1] - pos2[1]) ** 2) ** 0.5
                    # check all robots
                    min_dist_of_inter_robot_to_any_robot = min(
                        min_dist_of_inter_robot_to_any_robot, min_inter_robot_dist_of_any)

                    # print for me
                    # if min_dist_of_inter_robot_to_any_robot < radius * 2:
                    # print("Collision with other robots at time:",
                    #   counter, " between robots", robot_id, "and", j, "with positions", pos1, pos2)
            # save for falsification
            list_min_inter_robot_dist_of_any_robot.append(
                min_dist_of_inter_robot_to_any_robot)

            local_min_inter_robot_dist_of_any_robot = min(local_min_inter_robot_dist_of_any_robot,
                                                          min_dist_of_inter_robot_to_any_robot)
            local_pos_list_for_inter_robot_dist_of_any_robot.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_inter_robot_dist_of_any_robot.append(
                    updated_position)

            """
            # ==========================================
            # decide navigating attack for victim robot
            # dist: difference of static point and victim's position
            print("Herding Attack")

            dist_victim_point = calculate_abs_distance(pointX, pointY,
                                                       updated_positions[victimRobotId * 2 + 0], updated_positions[victimRobotId * 2 + 1])
            # save for falsification
            list_min_navigating_to_point_victim.append(dist_victim_point)

            # save to print
            local_dist_victim = min(local_dist_victim, dist_victim_point)
            local_pos_list_for_dist_victim.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_dist_victim.append(updated_position)

            """
            # ==========================================
            # decide navigating attack for any robot
            # dist: difference of static point and victim's position
            dist_any_point = float('inf')
            for robot_id, robot_pos in enumerate(updated_positions):
                if robot_id == attackedRobotId:
                    continue
                dist_any_point = min(dist_any_point, calculate_abs_distance(pointX, pointY,
                                                                            updated_positions[robot_id][0], updated_positions[robot_id][1]))
            # list for falsification
            list_min_navigating_to_point_any.append(dist_any_point)

            # save to print
            local_dist_any = min(local_dist_any, dist_any_point)
            local_pos_list_for_dist_any.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_dist_any.append(updated_position)
            """

            # ==========================================
            # decide deadlock of victim robot:
            print("Deadlock Attack")

            last_positions.append(
                updated_positions[victimRobotId * 2: victimRobotId * 2 + 2])
            # Keep only the last positions
            if len(last_positions) > deadlockTimestep:
                last_positions.pop(0)

            total_pos_change = 0.0
            if len(last_positions) == deadlockTimestep:
                # sum of the move changes between consecutive positions in last_positions
                for i in range(len(last_positions) - 1):
                    total_pos_change += calculate_abs_distance(
                        last_positions[i][0], last_positions[i][1], last_positions[i+1][0], last_positions[i+1][1])
            else:
                total_pos_change = float('inf')

            # edge case
            # skip last moments if the robot is close to the goal within 10 meters and does not move more.
            if calculate_abs_distance(updated_positions[victimRobotId * 2], updated_positions[victimRobotId * 2 + 1], goal_positions[victimRobotId][0], goal_positions[victimRobotId][1]) < 10.0:
                total_pos_change = float('inf')

            # print for me
            # if total_pos_change < deadlockPosChange:
                # print("Deadlock of victim robot at time:", counter, " with position",
                #   updated_positions[victimRobotId], " and measurement: ", total_pos_change)

            # save for falsification
            list_min_deadlock_of_victim_robot.append(
                total_pos_change)

            # save to print
            local_min_deadlock_of_victim_robot = min(
                local_min_deadlock_of_victim_robot, total_pos_change)
            local_pos_list_for_deadlock_of_victim_robot.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_deadlock_of_victim_robot.append(
                    updated_position)

            """
            # ==========================================
            # decide deadlock of any robot:
            min_deadlock_measurement = float('inf')
            for robot_id in range(num_of_robots):
                # skip attacked robot
                if robot_id == attackedRobotId:
                    continue
                # Append the current position of each robot
                last_positions_dict[robot_id].append(
                    updated_positions[robot_id])

                # Keep only the last 'deadlockTimestep' positions for each robot
                if len(last_positions_dict[robot_id]) > deadlockTimestep:
                    last_positions_dict[robot_id].pop(0)

                # Calculate deadlock measurement if enough positions are recorded
                if len(last_positions_dict[robot_id]) == deadlockTimestep:
                    # edge case
                    # skip last moments if the robot is close to the goal within 10 meters
                    if calculate_abs_distance(updated_positions[robot_id][0], updated_positions[robot_id][1], goal_positions[robot_id][0], goal_positions[robot_id][1]) < 10.0:
                        distance_moved = float('inf')
                    else:
                        distance_moved = 0.0
                        for i in range(len(last_positions_dict[robot_id]) - 1):
                            distance_moved += calculate_abs_distance(
                                last_positions_dict[robot_id][i][0], last_positions_dict[robot_id][i][1],
                                last_positions_dict[robot_id][i+1][0], last_positions_dict[robot_id][i+1][1])
                    min_deadlock_measurement = min(
                        min_deadlock_measurement, distance_moved)
                else:
                    min_deadlock_measurement = float('inf')

            # print for me
            # if min_deadlock_measurement < deadlockPosChange:
                # print("Deadlock of victim robot at time:", counter)

            # save for falsification
            list_min_deadlock_of_any_robot.append(min_deadlock_measurement)

            # save to print
            local_min_deadlock_of_any_robot = min(
                local_min_deadlock_of_any_robot, min_deadlock_measurement)
            local_pos_list_for_deadlock_of_any_robot.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_deadlock_of_any_robot.append(
                    updated_position)
            """
            # ==========================================
            # decide navigation delay of victim robot:
            print("Navigation Delay Attack")
            current_position = [updated_positions[victimRobotId * 2 + 0],
                                updated_positions[victimRobotId * 2 + 1]]
            remaining_distance = calculate_abs_distance(
                current_position[0], current_position[1], goal_positions[victimRobotId][0], goal_positions[victimRobotId][1])

            # save for falsification
            list_min_navigation_duration_of_victim_robot.append(
                remaining_distance)
            # print("here!")

            # save to print
            local_min_navigation_duration_of_victim_robot = min(
                local_min_navigation_duration_of_victim_robot, remaining_distance)
            local_pos_list_for_navigation_duration_of_victim_robot.append(
                counter)
            for updated_position in updated_positions:
                local_pos_list_for_navigation_duration_of_victim_robot.append(
                    updated_position)

            # ==========================================
            # decide navigation delay of any robot:
            # TODO not working, it should be the same robot
            """
            remaining_distance_any = -float('inf')
            temp_robot_id = -1

            for robot_id in range(num_of_robots):
                if robot_id == attackedRobotId:
                    continue

                # navigation delay
                current_position = updated_positions[robot_id]
                # distance to goal
                rem_distance = calculate_abs_distance(
                    current_position[0], current_position[1], goal_positions[robot_id][0], goal_positions[robot_id][1])

                if rem_distance > remaining_distance_any:
                    remaining_distance_any = rem_distance
                    temp_robot_id = robot_id
            # max distanced robot is decided.

            # now check its deadlock
            # Calculate deadlock measurement if enough positions are recorded
            distance_moved_for_nav_dur = float('inf')
            if len(last_positions_dict[temp_robot_id]) == deadlockTimestep:
                # edge case
                # skip last moments if the robot is close to the goal within 10 meters
                if calculate_abs_distance(updated_positions[temp_robot_id][0], updated_positions[temp_robot_id][1], goal_positions[temp_robot_id][0], goal_positions[temp_robot_id][1]) < 10.0:
                    distance_moved_for_nav_dur = float('inf')
                else:
                    # calculate the distance moved
                    distance_moved_for_nav_dur = 0.0
                    for i in range(len(last_positions_dict[temp_robot_id]) - 1):
                        distance_moved_for_nav_dur += calculate_abs_distance(
                            last_positions_dict[temp_robot_id][i][0], last_positions_dict[temp_robot_id][i][1],
                            last_positions_dict[temp_robot_id][i+1][0], last_positions_dict[temp_robot_id][i+1][1])
            else:
                distance_moved_for_nav_dur = float('inf')
            # robot's deadlock is decided. Now, check if it is the minimum

            # save for falsification
            list_max_navigation_duration_of_any_robot.append(
                remaining_distance_any)
            list_min_deadlock_for_navigation_duration_of_any_robot.append(
                distance_moved_for_nav_dur)

            # save to print
            local_max_navigation_duration_of_any_robot = max(
                local_max_navigation_duration_of_any_robot, remaining_distance_any)
            local_pos_list_for_navigation_duration_of_any_robot.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_navigation_duration_of_any_robot.append(
                    updated_position)
            """

            # ==========================================
            temp_list_for_global.append(counter)
            temp_list_for_global_with_velocities.append(counter)
            for updated_position in updated_positions:
                temp_list_for_global.append(updated_position)
                temp_list_for_global_with_velocities.append(updated_position)
            # for updated_velocity in updated_velocities:
                # temp_list_for_global_with_velocities.append(updated_velocity)

            counter += glas_instance.param.sim_dt

        if attackType == 1:
            # victim - static obstacle collision
            if local_min_obstacle_dist_of_victim < global_min_dist_of_victim_to_static_so_far:
                global_min_dist_of_victim_to_static_so_far = local_min_obstacle_dist_of_victim
                global_log_list = local_pos_list_for_obstacle_dist_of_victim
        elif attackType == 2:
            # any robot - static obstacle collision
            if local_min_obstacle_dist_of_any_robot < global_min_dist_of_any_to_static_so_far:
                global_min_dist_of_any_to_static_so_far = local_min_obstacle_dist_of_any_robot
                global_log_list = local_pos_list_for_obstacle_dist_of_any_robot
        elif attackType == 3:
            # victim robot - other robots collision
            if local_min_inter_robot_dist_of_victim_robot < global_min_dist_of_victim_to_robots_so_far:
                global_min_dist_of_victim_to_robots_so_far = local_min_inter_robot_dist_of_victim_robot
                global_log_list = local_pos_list_for_inter_robot_dist_of_victim_robot
        elif attackType == 4:
            # any robot - other robots collision
            if local_min_inter_robot_dist_of_any_robot < global_min_dist_of_any_to_robots_so_far:
                global_min_dist_of_any_to_robots_so_far = local_min_inter_robot_dist_of_any_robot
                global_log_list = local_pos_list_for_inter_robot_dist_of_any_robot
        elif attackType == 5:
            # navigating the point - victim robot
            if local_dist_victim < global_min_navigating_to_point_victim:
                global_min_navigating_to_point_victim = local_dist_victim
                global_log_list = local_pos_list_for_dist_victim
        elif attackType == 6:
            # navigating the point - any robot
            if local_dist_any < global_min_navigating_to_point_any:
                global_min_navigating_to_point_any = local_dist_any
                global_log_list = local_pos_list_for_dist_any
        elif attackType == 7:
            # victim robot - deadlock
            if local_min_deadlock_of_victim_robot < global_min_deadlock_of_victim_robot:
                global_min_deadlock_of_victim_robot = local_min_deadlock_of_victim_robot
                global_log_list = local_pos_list_for_deadlock_of_victim_robot
        elif attackType == 8:
            # any robot - deadlock
            if local_min_deadlock_of_any_robot < global_min_deadlock_of_any_robot:
                global_min_deadlock_of_any_robot = local_min_deadlock_of_any_robot
                global_log_list = local_pos_list_for_deadlock_of_any_robot
        elif attackType == 9:
            # victim robot - navigation delay
            if local_min_navigation_duration_of_victim_robot < global_min_navigation_duration_of_victim_robot:
                global_min_navigation_duration_of_victim_robot = local_min_navigation_duration_of_victim_robot
                global_log_list = local_pos_list_for_navigation_duration_of_victim_robot
        elif attackType == 10:
            # any robot - navigation delay
            if local_max_navigation_duration_of_any_robot > global_max_navigation_duration_of_any_robot:
                global_max_navigation_duration_of_any_robot = local_max_navigation_duration_of_any_robot
                global_log_list = local_pos_list_for_navigation_duration_of_any_robot

    except Exception as e:
        print("Errors related to falsification!")
        print(e)
    try:
        print("time_result:", time_result, len(time_result))
        print("list_min_obstacle_dist_of_victim:",
              list_min_obstacle_dist_of_victim)
        print("list_min_obstacle_dist_of_any_robot:",
              list_min_obstacle_dist_of_any_robot)
        print("list_min_inter_robot_dist_of_victim_robot:",
              list_min_inter_robot_dist_of_victim_robot)
        print("list_min_inter_robot_dist_of_any_robot:",
              list_min_inter_robot_dist_of_any_robot)
        print("list_min_navigating_to_point_victim:",
              list_min_navigating_to_point_victim)
        print("list_min_navigating_to_point_any:",
              list_min_navigating_to_point_any)
        print("list_min_deadlock_of_victim_robot:",
              list_min_deadlock_of_victim_robot)
        print("list_min_deadlock_of_any_robot:",
              list_min_deadlock_of_any_robot)
        print("list_min_navigation_duration_of_victim_robot:", list_min_navigation_duration_of_victim_robot, len(
            list_min_navigation_duration_of_victim_robot))
        print("list_max_navigation_duration_of_any_robot:",
              list_max_navigation_duration_of_any_robot)
        print("list_min_deadlock_for_navigation_duration_of_any_robot:",
              list_min_deadlock_for_navigation_duration_of_any_robot)

        """
        trace = Trace(time_result, [time_result,
                                    list_min_obstacle_dist_of_victim,
                                    list_min_obstacle_dist_of_any_robot,
                                    list_min_inter_robot_dist_of_victim_robot,
                                    list_min_inter_robot_dist_of_any_robot,
                                    list_min_navigating_to_point_victim,
                                    list_min_navigating_to_point_any,
                                    list_min_deadlock_of_victim_robot,
                                    list_min_deadlock_of_any_robot,
                                    list_min_navigation_duration_of_victim_robot,
                                    list_max_navigation_duration_of_any_robot,
                                    list_min_deadlock_for_navigation_duration_of_any_robot])
        """
        # TODO: hacker
        trace = Trace(time_result, [time_result,
                                    list_min_obstacle_dist_of_victim,
                                    # [0.0] * len(time_result),
                                    [0.0] * len(time_result),
                                    list_min_inter_robot_dist_of_victim_robot,
                                    # [0.0] * len(time_result),
                                    [0.0] * len(time_result),
                                    list_min_navigating_to_point_victim,
                                    [0.0] * len(time_result),
                                    # [0.0] * len(time_result),
                                    list_min_deadlock_of_victim_robot,
                                    [0.0] * len(time_result),
                                    # [0.0] * len(time_result),
                                    list_min_navigation_duration_of_victim_robot,
                                    [0.0] * len(time_result),
                                    [0.0] * len(time_result)])
        global_log_list_last = temp_list_for_global
        global_log_list_prev = temp_list_for_global_prev
        global_log_list_fake = temp_list_for_global_fake
        global_log_list_last_with_velocities = temp_list_for_global_with_velocities
        # print("global_log_list_last: ", global_log_list_last)
        # print("result:")
        # print(trace)
        return BasicResult(trace)
    except Exception as e:
        print("Exception occurred during simulation")
        return FailureResult()


state_matcher = {
    "t": 0,
    "min_dist_of_victim_to_obstacles": 1,
    "min_dist_of_any_robot_to_obstacles": 2,
    "min_of_inter_robot_dis_of_victim_robot": 3,
    "min_of_inter_robot_dis_of_any_robot": 4,
    "min_dist_victim": 5,
    "min_dist_any": 6,
    "deltaPos_VictimRobot": 7,
    "deltaPos_AnyRobot": 8,
    "distToGoal_VictimRobot": 9,
    "distToGoal_AnyRobot": 10,
    "deltaPos_Any_for_NavigationDelay": 11
}

# Decide attack type
specification = None
attackName = ""
if attackType == 1:
    attackName = "collision_obstacle_victim_robot"
    phi_collision_obstacle_victim_drone = f"always (min_dist_of_victim_to_obstacles > {radius})"
    print("phi_collision_obstacle_targeted_drone: ",
          phi_collision_obstacle_victim_drone)
    specification = RTAMTDense(
        phi_collision_obstacle_victim_drone, state_matcher)
elif attackType == 2:
    attackName = "collision_obstacle_any_robot"
    phi_collision_obstacle_untargeted_drone = f"always (min_dist_of_any_robot_to_obstacles > {radius})"
    print("phi_collision_static_untargeted_drone: ",
          phi_collision_obstacle_untargeted_drone)
    specification = RTAMTDense(
        phi_collision_obstacle_untargeted_drone, state_matcher)
elif attackType == 3:
    attackName = "collision_btw_victim_and_other_robots"
    phi_collision_btw_victim_and_other_robots = f"always (min_of_inter_robot_dis_of_victim_robot > {radius*2})"
    print("phi_collision_btw_victim_and_other_robots: ",
          phi_collision_btw_victim_and_other_robots)
    specification = RTAMTDense(
        phi_collision_btw_victim_and_other_robots, state_matcher)
elif attackType == 4:
    attackName = "collision_btw_any_and_other_robots"
    phi_collision_btw_any_and_other_robots = f"always (min_of_inter_robot_dis_of_any_robot > {radius*2})"
    print("phi_collision_btw_any_and_other_robots: ",
          phi_collision_btw_any_and_other_robots)
    specification = RTAMTDense(
        phi_collision_btw_any_and_other_robots, state_matcher)
elif attackType == 5:
    attackName = "navigating_to_point_targeted"
    phi_navigating_robot_targeted = "always (min_dist_victim > 1.0)"
    print("phi_navigating_victim: ", phi_navigating_robot_targeted)
    specification = RTAMTDense(
        phi_navigating_robot_targeted, state_matcher)
elif attackType == 6:
    attackName = "navigating_to_point_untargeted"
    phi_navigating_robot_untargeted = "always (min_dist_any > 1.0)"
    print("phi_navigating_victim: ", phi_navigating_robot_untargeted)
    specification = RTAMTDense(
        phi_navigating_robot_untargeted, state_matcher)
elif attackType == 7:
    attackName = "deadlock_of_victim_robot"
    phi_deadlock_of_victim_robot = f"always deltaPos_VictimRobot > {deadlockPosChange}"
    print("phi_deadlock_of_victim_robot: ", phi_deadlock_of_victim_robot)
    specification = RTAMTDense(
        phi_deadlock_of_victim_robot, state_matcher)
elif attackType == 8:
    attackName = "deadlock_of_any_robot"
    phi_deadlock_of_any_robot = f"always deltaPos_AnyRobot > {deadlockPosChange}"
    print("phi_deadlock_of_any_robot: ", phi_deadlock_of_any_robot)
    specification = RTAMTDense(
        phi_deadlock_of_any_robot, state_matcher)
elif attackType == 9:
    attackName = "navigation_delay_of_victim_robot"
    # phi_navigation_delay_of_victim_robot = f"eventually (distToGoal_VictimRobot <= 3.0) or (eventually deltaPos_VictimRobot <= {deadlockPosChange})"
    phi_navigation_delay_of_victim_robot = f"eventually (distToGoal_VictimRobot <= 1.0)"
    print("phi_navigation_delay_of_victim_robot: ",
          phi_navigation_delay_of_victim_robot)
    specification = RTAMTDense(
        phi_navigation_delay_of_victim_robot, state_matcher)
elif attackType == 10:
    attackName = "navigation_delay_of_any_robot"
    phi_navigation_delay_of_any_robot = f"eventually (distToGoal_AnyRobot <= 3.0) or (eventually deltaPos_Any_for_NavigationDelay <= {deadlockPosChange})"
    print("phi_navigation_delay_of_any_robot: ",
          phi_navigation_delay_of_any_robot)
    specification = RTAMTDense(
        phi_navigation_delay_of_any_robot, state_matcher)
else:
    print("Invalid attack type!")
    exit()


print(time_step, maxSpeed, totalTimeStep)
# message spoofing variables
initial_conditions = [
    (0, 50),
    # (time_step * maxSpeed * -1, time_step * maxSpeed),
    # (time_step * maxSpeed * -1, time_step * maxSpeed),
    (-5, 5),
    (-5, 5)
    # (maxSpeed * -1, maxSpeed),
    # (maxSpeed * -1, maxSpeed)
] * numberOfFalseMessage

env_file_name = "/Users/doguhanyeke/Desktop/research/swarm/falsi_glas/results/singleintegrator/instances/map_8by8_obst12_agents8_ex0000.yaml"
obstacles = load_obstacles_from_yaml(env_file_name)
print("Obstacles: ", obstacles)
goal_positions = load_goals_from_yaml(env_file_name)
print("Goal Positions: ", goal_positions)

options = Options(runs=falsificationRuns, iterations=falsificationIterations, interval=(
    0, 10), static_parameters=initial_conditions)

optimizer = DualAnnealing()

result = staliro(Glas_Model, specification, optimizer, options)

worst_run_ = worst_run(result)
worst_sample = worst_eval(worst_run_).sample
worst_result = simulate_model(Glas_Model, options, worst_sample)

print("\nWorst Sample:")
print(worst_sample)

print("\nResult:")
print(worst_result.trace.states)

print("\nWorst Evaluation:")
print(worst_eval(worst_run(result)))

# sample_xs = [evaluation.sample[0] for evaluation in worst_run_.history]
# sample_ys = [evaluation.sample[1] for evaluation in worst_run_.history]

with open(f'/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/falsiResults2/{attackName}_run_positions_{falsificationIterations}ite_{numberOfFalseMessage}mes_{environment_name}.txt', 'w') as file:
    for i in range(0, len(global_log_list), num_of_robots * 2 + 1):
        time = global_log_list[i]
        positions = global_log_list[i+1:i+num_of_robots*2+1]
        positions_str = ' '.join(
            f"({positions[ind]:.4f},{positions[ind+1]:.4f})" for ind in range(0, len(positions), 2)
        )
        file.write(f"{time} {positions_str}\n")
print("End")

f_falsiresult = open(
    "/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/result_attack.txt", "w")
# if evaluation cost is negative, then print
if worst_eval(worst_run_).cost < 0:
    f_falsiresult.write("1")
    print("Evaluation cost: ", worst_eval(worst_run_).cost)
    with open(f'/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/falsiResults3/{attackName}_run_positions_{falsificationIterations}ite_{numberOfFalseMessage}mes_{environment_name}_glas.txt', 'w') as file:
        for i in range(0, len(global_log_list_last), num_of_robots * 2 + 1):
            time = global_log_list_last[i]
            positions = global_log_list_last[i+1:i+num_of_robots*2+1]
            positions_str = ' '.join(
                f"({positions[ind]:.4f},{positions[ind+1]:.4f})" for ind in range(0, len(positions), 2)
            )
        file.write(f"{time} {positions_str}\n")
    with open(f'/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/falsiResults5/{attackName}_run_positions_prev_{falsificationIterations}ite_{numberOfFalseMessage}mes_{environment_name}_prev_glas.txt', 'w') as file:
        for i in range(0, len(global_log_list_prev), num_of_robots * 2 + 1):
            time = global_log_list_prev[i]
            positions = global_log_list_prev[i+1:i+num_of_robots*2+1]
            positions_str = ' '.join(
                f"({positions[ind]:.4f},{positions[ind+1]:.4f})" for ind in range(0, len(positions), 2)
            )
        file.write(f"{time} {positions_str}\n")
    with open(f'/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/falsiResults5/{attackName}_run_positions_fake_{falsificationIterations}ite_{numberOfFalseMessage}mes_{environment_name}_fake_glas.txt', 'w') as file:
        for i in range(0, len(global_log_list_fake), num_of_robots * 2 + 1):
            time = global_log_list_fake[i]
            positions = global_log_list_fake[i+1:i+num_of_robots*2+1]
            positions_str = ' '.join(
                f"({positions[ind]:.4f},{positions[ind+1]:.4f})" for ind in range(0, len(positions), 2)
            )
        file.write(f"{time} {positions_str}\n")
    print("End2")
    # with open(f'/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/falsiResults4/{attackName}_run_positions_{falsificationIterations}ite_{numberOfFalseMessage}mes_{environment_name}.txt', 'w') as file:
    #     for i in range(0, len(global_log_list_last_with_velocities), 2 * num_of_robots + 1):
    #         time = global_log_list_last_with_velocities[i]
    #         positions = [
    #             global_log_list_last_with_velocities[i+1:i+num_of_robots+1]]
    #         positions = positions[0]
    #         # print("positions: ", positions)
    #         positions_str = ' '.join(
    #             [f"{pos[0]:.4f} {pos[1]:.4f}" for pos in positions])
    #         # print("positions_str: ", positions_str)
    #         velocities = global_log_list_last_with_velocities[i +
    #                                                           num_of_robots+1:i+2*num_of_robots+1]
    #         # print("velocities: ", velocities)
    #         velocities_str = ' '.join(
    #             [f"{vel[0]:.4f} {vel[1]:.4f}" for vel in velocities])
    #         # print("velocities_str: ", velocities_str)
    #         file.write(f"{time} {positions_str} {velocities_str}\n")
else:
    f_falsiresult.write("0")
f_falsiresult.close()

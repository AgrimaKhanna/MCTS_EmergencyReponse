import pickle
from itertools import chain
import os
import random
from collections import namedtuple
import pandas as pd
import numpy as np
from math import sqrt, log

# Define the State of the emergency response system
class State:
    def __init__(self, ambulance_position, incident_location):
        self.ambulance_position = ambulance_position
        self.incident_location = incident_location

    def move_ambulance(self, direction):
        x, y = self.ambulance_position
        if direction == 'up':
            y -= 1
        elif direction == 'down':
            y += 1
        elif direction == 'left':
            x -= 1
        elif direction == 'right':
            x += 1
        # Ensure the ambulance stays within the bounds of the grid
        x = max(0, min(x, 29))
        y = max(0, min(y, 29))
        return State((x, y), self.incident_location)

    def is_at_incident(self):
        return self.ambulance_position == self.incident_location

# Define the Node in the MCTS
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = -1  # Wins are now negative rewards
        self.visits = 0
        self.untried_actions = ['up', 'down', 'left', 'right']

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def add_child(self, action):
        new_state = self.state.move_ambulance(action)
        child_node = Node(new_state, parent=self)
        self.children.append(child_node)
        if action in self.untried_actions:  # Ensure the action is in the list before removing
            self.untried_actions.remove(action)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.wins += reward  # Wins are negative rewards, so we add

    def tree_to_string(self, level=0):
        ret = "\t" * level + f"Node: Pos={self.state.ambulance_position}, Wins={self.wins}, Visits={self.visits}\n"
        for child in self.children:
            ret += child.tree_to_string(level+1)
        return ret

    def is_leaf(self):
        return len(self.children) == 0  # A node is a leaf if it has no children


def save_tree_to_file(node, filename):
    with open(filename, 'w') as file:
        tree_str = node.tree_to_string()
        file.write(tree_str)
        print(f"The tree has been saved to {filename}")

# Define the MCTS algorithm
class MCTS():
    def __init__(self, state):
        self.root = Node(state)

    def selection(self, node, C = 1.4):
        # Select the child with the highest UCB1 score
        best_score = float('-inf')
        best_child = None
        for child in node.children:
            # Note: wins are negative, so we subtract to maximize score
            exploit = child.wins / child.visits
            explore = C * sqrt(log(node.visits) / child.visits)
            score = -exploit + explore  # Minimize negative reward (maximize score)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expansion(self, node):
        if not node.untried_actions:  # Check if there are no more actions to try
            return node  # In this case, simply return the node itself

        action = node.untried_actions.pop()  # Remove an action from the list
        return node.add_child(action)

    def simulation(self, node):
        current_state = node.state
        reward = 0
        incident_reward = 100  # High positive reward for reaching the incident
        while not current_state.is_at_incident():
            if len(node.untried_actions) == 0:
                break  # No more actions to try
            action = random.choice(node.untried_actions)  # Choose a random action
            current_state = current_state.move_ambulance(action)
            reward -= 1  # Each step costs -1

        if current_state.is_at_incident():
            reward += incident_reward  # Add reward for reaching the incident

        return reward

    def backpropagation(self, node, reward):
        # Propagate the simulation results back up the tree
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_move(self):
        # Choose the child of the root with the least negative wins (highest score)
        best_score = float('inf')
        best_move = None
        for child in self.root.children:
            score = child.wins / child.visits
            if score < best_score:
                best_score = score
                best_move = child.state.ambulance_position
        return best_move

    def find_best_leaf(self):
        best_score = float('-inf')
        best_leaf = None

        def evaluate_leaf(node):
            nonlocal best_score, best_leaf
            if node.is_leaf():
                score = node.wins / node.visits  # Calculate the score for this leaf
                if score > best_score:
                    best_score = score
                    best_leaf = node
            else:
                for child in node.children:
                    evaluate_leaf(child)

        evaluate_leaf(self.root)
        return best_leaf

    def run(self, max_iterations):
        for _ in range(max_iterations):
            leaf = self.traverse(self.root)  # Start from the root and traverse the tree to find the best leaf to explore
            reward = self.simulation(leaf)   # Simulate a playout from the leaf
            self.backpropagation(leaf, reward)  # Update the tree with the simulation results

    def traverse(self, node):
        while node.is_fully_expanded():
            C = 0.7  # Exploration parameter
            node = self.selection(node, C)
        return self.expansion(node) if not node.is_fully_expanded() else node


# Function to trace back from the best leaf node to the root
def trace_best_path(leaf_node):
    path = []
    current_node = leaf_node
    while current_node.parent is not None:  # Trace back to the root
        path.append(current_node)
        current_node = current_node.parent
    return path[::-1]  # Reverse the path to start from the root


def generate_random_ambulance_positions(num_ambulances, grid_size):
    return [divmod(random.randint(0, grid_size**2 - 1), grid_size) for _ in range(num_ambulances)]

# Function to randomly sample an incident from the DataFrame
def sample_incident(dataframe):
    sampled_incident = dataframe.sample(1).iloc[0]
    incident_location = divmod(sampled_incident['ID'], 30)
    return incident_location

if __name__ == "__main__":
    directory = 'data'
    incident_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pickle')]

    incident_frames = []
    for file in incident_files:
        s = pd.read_pickle(file)
        flattened_list = list(chain.from_iterable(s))

        incident = pd.DataFrame(flattened_list, columns=['ID', 'DateTime'])
        incident_frames.append(incident)

    incidents = pd.concat(incident_frames, ignore_index=True)

    # Randomly sample one incident from the data
    incident_location = sample_incident(incidents)
    print("Incident reported at location:", incident_location)

    # Generate random initial positions for the ambulances
    num_ambulances = 4
    grid_size = 30
    ambulance_positions = generate_random_ambulance_positions(num_ambulances, grid_size)
    print("Ambulance positions:")
    for i, position in enumerate(ambulance_positions):
        print("Ambulance", i, "position:", position)
    # Run MCTS for each ambulance and record the best score and move
    best_score = float('inf')
    best_ambulance = None
    best_move = None

    for index, position in enumerate(ambulance_positions):
        initial_state = State(position, incident_location)
        mcts = MCTS(initial_state)
        mcts.run(5000)  # Run the MCTS for a fixed number of iterations
        score = mcts.root.wins / mcts.root.visits  # Get the score of the root node
        save_tree_to_file(mcts.root, f'mcts_tree_ambulance_{index}.txt')

        if score < best_score:  # Check if this is the best score so far
            best_score = score
            best_ambulance = index
            best_move = mcts.best_move()  # This is the best move for this ambulance
            best_leaf = mcts.find_best_leaf()
            best_path = trace_best_path(best_leaf)

            # Select the ambulance with the best score
    print("Dispatch ambulance number", best_ambulance, "from position", ambulance_positions[best_ambulance])
    print("Best path for the selected ambulance:")
    for node in best_path:
        print(f"Node: Position={node.state.ambulance_position}, Wins={node.wins}, Visits={node.visits}")

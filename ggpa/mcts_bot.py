from __future__ import annotations
import math
from copy import deepcopy
import time
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose
import random


# You only need to modify the TreeNode!
class TreeNode:
    # You can change this to include other attributes. 
    # param is the value passed via the -p command line option (default: 0.5)
    # You can use this for e.g. the "c" value in the UCB-1 formula
    def __init__(self, param, parent=None, action_from_parent=None):
        self.children = {}  # Maps action keys to (action, node) tuples
        self.parent = parent
        self.param = param
        self.visits = 0
        self.total_score = 0
        self.untried_actions = None  # Cache for unexplored actions
        self.action_from_parent = action_from_parent  # The action that led to this node
    
    # REQUIRED function
    # Called once per iteration
    def step(self, state):
        node, rollout_state = self.select(state)
        if not rollout_state.ended():
            node = node.expand(rollout_state)
        result = node.rollout(rollout_state)
        node.backpropagate(result)
        
    # REQUIRED function
    # Called after all iterations are done; should return the 
    # best action from among state.get_actions()
    def get_best(self, state):
        if not self.children:
            return random.choice(state.get_actions())
        
        # Filter out invalid actions
        valid_children = {
            k: v for k, v in self.children.items() 
            if v[0].to_action(state) is not None
        }
        
        if not valid_children:
            return random.choice(state.get_actions())
        
        # Return the action with the highest average score
        best_action = max(valid_children.values(), 
                         key=lambda x: x[1].total_score / (x[1].visits + 1e-6))
        return best_action[0]
        
    # REQUIRED function (implementation optional, but *very* helpful for debugging)
    # Called after all iterations when the -v command line parameter is present
    def print_tree(self, indent=0):
        for action_key, (action, child) in self.children.items():
            avg_score = child.total_score / (child.visits + 1e-6)
            print(" " * indent + f"{action}: {avg_score:.2f} ({child.visits} visits)")
            child.print_tree(indent + 2)

    # RECOMMENDED: select gets all actions available in the state it is passed
    # If there are any child nodes missing (i.e. there are actions that have not 
    # been explored yet), call expand with the available options
    # Otherwise, pick a child node according to your selection criterion (e.g. UCB-1)
    # apply its action to the state and recursively call select on that child node.
    def select(self, state):
        node = self
        while node.is_fully_expanded(state) and node.children and not state.ended():
            node = node.best_uct_child(state)
            state.step(node.action_from_parent)
        return node, state

    def is_fully_expanded(self, state):
        if self.untried_actions is None:
            self.untried_actions = state.get_actions()
            # Remove already expanded actions
            self.untried_actions = [a for a in self.untried_actions if str(a) not in self.children]
        return len(self.untried_actions) == 0

    # RECOMMENDED: expand takes the available actions, and picks one at random,
    # adds a child node corresponding to that action, applies the action ot the state
    # and then calls rollout on that new node
    def expand(self, state):
        # Pick one unexplored action
        if self.untried_actions is None:
            self.untried_actions = state.get_actions()
            self.untried_actions = [a for a in self.untried_actions if str(a) not in self.children]
        while self.untried_actions:
            action = self.untried_actions.pop()
            if action.to_action(state) is not None:
                next_state = deepcopy(state)
                next_state.step(action)
                child = TreeNode(self.param, parent=self, action_from_parent=action)
                self.children[str(action)] = (action, child)
                return child
        return self  # fallback if no valid actions

    # RECOMMENDED: rollout plays the game randomly until its conclusion, and then 
    # calls backpropagate with the result you get 
    def rollout(self, state):
        depth = 0
        max_depth = 20
        while not state.ended() and depth < max_depth:
            actions = state.get_actions()
            valid_actions = [a for a in actions if a.to_action(state) is not None]
            if not valid_actions:
                break
            # Heuristic: prefer block if player is low, attack if enemy is low, else prefer attack, then block, then random
            player_health = state.health() if hasattr(state, 'health') else 1.0
            # Estimate enemy health (average of all enemies)
            enemy_health = 1.0
            if hasattr(state, 'enemies') and state.enemies:
                total = sum(e.health for e in state.enemies)
                max_total = sum(e.max_health for e in state.enemies)
                if max_total > 0:
                    enemy_health = total / max_total
            attack = [a for a in valid_actions if hasattr(a, 'card') and a.card and ('Attack' in str(a.card) or 'Strike' in str(a.card[0]) or 'Bludgeon' in str(a.card[0]) or 'Searing Blow' in str(a.card[0]))]
            block = [a for a in valid_actions if hasattr(a, 'card') and a.card and ('Defend' in str(a.card) or 'Block' in str(a.card[0]) or 'Shrug it Off' in str(a.card[0]))]
            # If about to die, prioritize block
            if player_health < 0.4 and block:
                action = random.choice(block)
            # If enemy is low, prioritize attack
            elif enemy_health < 0.4 and attack:
                action = random.choice(attack)
            # Otherwise, prefer attack, then block, then random
            elif attack:
                action = random.choice(attack)
            elif block:
                action = random.choice(block)
            else:
                action = random.choice(valid_actions)
            state.step(action)
            depth += 1
            # Early termination if player is dead
            if hasattr(state, 'health') and state.health() <= 0:
                break
        return self.score(state)
        
    def best_uct_child(self, state):
        # UCB1 formula
        valid_children = [v for v in self.children.values() if v[0].to_action(state) is not None]
        if not valid_children:
            return list(self.children.values())[0][1]  # fallback
        log_N = math.log(self.visits + 1)
        def uct(child):
            node = child[1]
            avg = node.total_score / (node.visits + 1e-6)
            return avg + self.param * math.sqrt(log_N / (node.visits + 1e-6))
        return max(valid_children, key=uct)[1]

    # RECOMMENDED: backpropagate records the score you got in the current node, and 
    # then recursively calls the parent's backpropagate as well.
    # If you record scores in a list, you can use sum(self.results)/len(self.results)
    # to get an average.
    def backpropagate(self, result):
        self.visits += 1
        self.total_score += result
        if self.parent:
            self.parent.backpropagate(result)
        
    # RECOMMENDED: You can start by just using state.score() as the actual value you are 
    # optimizing; for the challenge scenario, in particular, you may want to experiment
    # with other options (e.g. squaring the score, or incorporating state.health(), etc.)
    def score(self, state):
        base_score = state.score()
        health_bonus = state.health() * 0.5 if hasattr(state, 'health') else 0
        # Penalize if game ended with player death
        if state.ended() and state.health() <= 0:
            return -1.0
        # Bonus for winning, scaled by how quickly
        if state.ended() and state.health() > 0:
            turn_bonus = 1.0 if not hasattr(state, 'turn') else max(0.5, 1.5 - 0.05 * state.turn)
            return 1.0 * turn_bonus + health_bonus
        # Penalize for taking too long
        if hasattr(state, 'turn') and state.turn > 10:
            return base_score * 0.5 + health_bonus
        # Penalize for low health
        if hasattr(state, 'health') and state.health() < 0.3:
            return base_score + health_bonus - 0.5
        return base_score + health_bonus
        
        
# You do not have to modify the MCTS Agent (but you can)
class MCTSAgent(GGPA):
    def __init__(self, iterations: int, verbose: bool, param: float):
        self.iterations = iterations
        self.verbose = verbose
        self.param = param

    # REQUIRED METHOD
    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        actions = battle_state.get_actions()
        if len(actions) == 1:
            return actions[0].to_action(battle_state)
    
        t = TreeNode(self.param)
        start_time = time.time()

        for i in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.step(sample_state)
        
        best_action = t.get_best(battle_state)
        if self.verbose:
            t.print_tree()
        
        if best_action is None:
            print("WARNING: MCTS did not return any action")
            return random.choice(self.get_choose_card_options(game_state, battle_state)) # fallback option
        return best_action.to_action(battle_state)
    
    # REQUIRED METHOD: All our scenarios only have one enemy
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]
    
    # REQUIRED METHOD: Our scenarios do not involve targeting cards
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]

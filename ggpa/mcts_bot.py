from __future__ import annotations
import math
from copy import deepcopy
import time
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from action.game_action import GameAction
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose
import random


# You only need to modify the TreeNode!
class TreeNode:
    # You can change this to include other attributes.
    # param is the value passed via the -p command line option (default: 0.5)
    # You can use this for e.g. the "c" value in the UCB-1 formula
    def __init__(self, param, parent=None):
        self.children = {}
        self.parent = parent
        self.results = []
        self.param = param
        self.visits = 0
        self.action = None

    # Helper to extract a consistent key for any GameAction
    def get_key(self, action):
        if hasattr(action, 'args'):
            return action.args
        if hasattr(action, 'card'):
            return action.card  # this is usually a (name, upgrade_count) tuple
        return (str(action), 0)  # dummy fallback


    # REQUIRED function
    # Called once per iteration
    def step(self, state):
        self.select(state)

    # REQUIRED function
    # Called after all iterations are done; should return the
    # best action from among state.get_actions()
    def get_best(self, state):
        best_score = -float('inf')
        best_action = None
        for key, child in self.children.items():
            action = GameAction(key)
            if not child.results:
                continue
            avg = sum(child.results) / len(child.results)
            if avg > best_score:
                best_score = avg
                best_action = action
        return best_action

    # REQUIRED function (implementation optional, but *very* helpful for debugging)
    # Called after all iterations when the -v command line parameter is present
    def print_tree(self, indent = 0):
        for key, child in self.children.items():
            action = GameAction(key)
            if not child.results:
                continue
            avg = sum(child.results) / len(child.results)
            print(" " * indent + f"{action} -> avg: {avg:.2f}, visits: {child.visits}")
            child.print_tree(indent + 2)

    # RECOMMENDED: select gets all actions available in the state it is passed
    def select(self, state):
        if state.ended():
            score = self.score(state)
            self.backpropagate(score)
            return

        options = state.get_actions()
        unexplored = []
        explored = []

        for action in options:
            key = self.get_key(action)
            if key not in self.children:
                unexplored.append(action)
            else:
                explored.append(action)

        if unexplored:
            option = random.choice(unexplored)
            self.expand(state, option)
            return

        best_score = -float('inf')
        best_action = None
        for action in explored:
            key = self.get_key(action)
            child = self.children[key]
            avg_score = sum(child.results) / len(child.results)
            ucb = avg_score + self.param * math.sqrt(math.log(self.visits) / child.visits)
            if ucb > best_score:
                best_score = ucb
                best_action = action

        key = self.get_key(best_action)
        state.step(best_action)
        self.children[key].select(state)

    # RECOMMENDED: expand adds a child node and performs rollout
    def expand(self, state, available):
        next_state = state.copy_undeterministic()
        next_state.step(available)

        key = self.get_key(available)
        child = TreeNode(self.param, parent=self)
        child.action = available
        self.children[key] = child

        child.rollout(next_state)

    # RECOMMENDED: rollout plays the game randomly until done, then backpropagates
    def rollout(self, state):
        while not state.ended():
            actions = state.get_actions()
            action = random.choice(actions)
            state.step(action)

        score = self.score(state)
        self.backpropagate(score)

    # RECOMMENDED: backpropagate result to this and parent
    def backpropagate(self, result):
        self.results.append(result)
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(result)

    # RECOMMENDED: score for a given state
    def score(self, state):
        #return state.score()
        # If the game is over and the enemy is dead, give full score
        if not state.enemies:
            return 1.0
        
        # Get player and enemy (only one enemy per scenario)
        player = state.player
        enemy = state.enemies[0]

        # Normalize HP ratios
        player_hp_ratio = player.health / player.max_health
        enemy_hp_ratio = enemy.health / enemy.max_health

        # Want high player health and low enemy health
        score = 0.5 * player_hp_ratio + 0.5 * (1.0 - enemy_hp_ratio)
        return score

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
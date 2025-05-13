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
from typing import TYPE_CHECKING
import random
if TYPE_CHECKING:
    from game import GameState
    from battle import BattleState
    from agent import Agent
    from card import Card


class Sampler:
    def __init__(self):
        self.results = {}
    def sample(self, state):
        actions = state.get_actions()
        if not actions:
            return 
        action = random.choice(actions)
        if action.key() not in self.results:
            self.results[action.key()] = []
        expansion_state = state.copy_undeterministic()
        expansion_state.step(action)
        score = self.rollout(expansion_state)
        self.results[action.key()].append(score)
            
    def rollout(self, state):
        while not state.ended():
            action = random.choice(state.get_actions())
            state.step(action)
        return state.score()
        
    def print_scores(self):
        for r in self.results:
            action = r if r else "End Turn"
            print("  ", action, sum(self.results[r])*1.0/len(self.results[r]))
        
    def get_best(self, options):
        best = None 
        best_score = 0
        for o in options:
            if o.key() in self.results:
                score = sum(self.results[o.key()])*1.0/len(self.results[o.key()])
                if score >= best_score:
                    best_score = score 
                    best = o
        return best
        
            
        
class SamplingAgent(GGPA):
    def __init__(self, seed: int, iterations: int, verbose: bool):
        self.iterations = iterations
        self.verbose = verbose
        self.random = random.Random(seed)

    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        t = Sampler()
        start_time = time.time()

        for i in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.sample(sample_state)
            
        if self.verbose:
            t.print_scores()
        
        best_action = t.get_best(battle_state.get_actions())

        return best_action.to_action(battle_state)
    
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]
    
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]
        
    def __deepcopy__(self, memo):
        result = SamplingAgent(0, self.iterations, self.verbose)
        result.random = deepcopy(self.random, memo)
        return result
        
        
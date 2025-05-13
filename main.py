from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import AcidSlimeSmall, SpikeSlimeSmall, JawWorm
from card import CardGen, CardRepo
import time
import agent
import random
from ggpa.human_input import HumanInput
from ggpa.backtrack import BacktrackBot
from ggpa.mcts_bot import MCTSAgent
from ggpa.random_bot import RandomAgent
from ggpa.sampling_bot import SamplingAgent
import argparse

def get_scenario(name):
    if name == "intro":
        return (20, ["Strike", "Strike", "Defend", "Defend", "Defend", "Thunderclap", "Thunderclap", "Thunderclap", "Inflame", "PommelStrike"], "Troll")
    if name == "offerings":
        return (19, ["Strike", "Offering", "Offering", "Offering", "Offering", "Offering", "Thunderclap", "Thunderclap", "SearingBlow"], "Troll")
    if name == "lowhp":
        return (8, ["Strike", "Offering", "Defend", "Defend", "Defend", "Defend", "Thunderclap", "Thunderclap", "Thunderclap", "Thunderclap"], "Troll")
    if name == "giant":
        return (16, ["Strike", "Bash", "Defend", "SearingBlow", "Bludgeon"], "Giant")
    if name == "challenge":
        return (8, ["Strike", "Bash", "Defend", "SearingBlow", "Bludgeon"], "Giant")
    if name == "boss":
        return (65, ["Strike", "Strike", "Defend", "Defend", "Bash", "Bludgeon", "Thunderclap", "Inflame", "PommelStrike", "Offering"], "Donut")

def main(scenario, n, verbose, bot, games, param, israndom):
    scores = []
    wins = 0
    agentname = ""
    for i in range(games):
        hp, deck, enemy = get_scenario(scenario)
        if bot == "mcts":
            agentname = "MCTS"
            player = MCTSAgent(n, verbose, param)
        elif bot == "random":
            agentname = "Random"
            player = RandomAgent()
        elif bot == "human":
            agentname = "Human"
            player = HumanInput(verbose)
        else:
            agentname = "Sampling"
            player = SamplingAgent(i, n, verbose)
        if not israndom:
            random.seed(i)
        game_state = GameState(Character.IRON_CLAD, player, 0, hp)
        game_state.set_deck(CardRepo.make_deck(deck))
        battle_state = BattleState(game_state, agent.make_enemy(enemy, game_state), verbose=Verbose.LOG if games <= 3 else Verbose.NO_LOG)
        start = time.time()
        battle_state.run()
        score = battle_state.score()
        if score > 0.999:
            wins += 1
        end = time.time()
        print(f"run ended in {end-start} seconds, score: {score}")
        scores.append(score)
    if games > 1:
        print(agentname, "average score:", sum(scores)*1.0/len(scores), "win rate:", "%.2f%%"%(wins*100.0/len(scores)))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='MiniStS',
                    description='A small version of the Slay the Spire combat loop')
    parser.add_argument('-n', '--iterations', type=int, default=50)
    parser.add_argument('-s', '--scenario', default="intro")      
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-b', '--bot', default="mcts")
    parser.add_argument('-g', '--games', type=int, default=1)
    parser.add_argument('-p', '--parameter', type=float, default=0.5)
    parser.add_argument('-r', '--random', action="store_true")
    args = parser.parse_args()
    main(args.scenario, args.iterations, args.verbose, args.bot, args.games, args.parameter, args.random)
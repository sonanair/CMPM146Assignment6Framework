from __future__ import annotations
from agent import Player
from config import Character
from card import CardRepo

class GameState:
    def __init__(self, character: Character, bot: GGPA, ascension: int, max_health=None):
        self.player = Player(character, bot, max_health)
        self.ascension = ascension
        self.deck: list[Card] = CardRepo.get_starter(character)
        self.draw_count = 5
        self.max_mana = 3

    def add_to_deck(self, *cards):
        self.deck.extend(cards)

    def set_deck(self, cards, *extracards):
        if isinstance(cards, list):
            self.deck = cards
        else:
            self.deck = [cards]
        self.deck.extend(extracards)

    def get_end_results(self):
        if self.player.is_dead():
            return -1
        return 1 # TODO is this a good idea? is this actually win?
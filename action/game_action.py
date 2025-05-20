from action.action import PlayCard,EndAgentTurn

class GameAction:
    def __init__(self, card=None):
        self.card = card
    def __eq__(self, other):
        return self.card == other.card
    def key(self):
        if self.card is None:
            return ""
        return f"{self.card[0]}:{self.card[1]}"
    def is_card(self, card):
        name,upgrade_count = self.card 
        return (name == card.name and upgrade_count == card.upgrade_count)
    def to_action(self, state):
        if self.card is None:
            return EndAgentTurn()
        for i in range(len(state.hand)):
            if self.is_card(state.hand[i]):
                return PlayCard(i)
        # If we can't find the card, return None to indicate invalid action
        return None
    def __str__(self):
        if self.card is None:
            return "End Turn"
        upgrades = "+"*self.card[1]
        return f"Play {self.card[0]}{upgrades}"
from framework import RewardModule, register_reward

@register_reward('tm_score')
class TMScoreReward(RewardModule):
    """
    Example reward module using TM-score for structure similarity.
    Replace/extend with other reward functions as needed.
    """
    def compute(self, state, action, next_state):
        # Compute TM-score or other reward for the transition
        pass 
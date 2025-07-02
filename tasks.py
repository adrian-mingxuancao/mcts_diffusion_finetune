from framework import TaskModule, register_task

@register_task('inverse_folding')
class InverseFoldingTask(TaskModule):
    """
    Example task module for protein inverse folding (structure → sequence).
    This version uses plDDT-based masking: scan the predicted sequence for positions with lowest plDDT and mask those for targeted improvement.
    Replace/extend with other tasks as needed.
    """
    def get_initial_state(self, input_data):
        # input_data: structure representation
        # Return initial state for search/planning
        pass
    def is_terminal(self, state):
        # Return True if state is a completed sequence
        pass
    def available_actions(self, state):
        # Return list of possible actions (e.g., which positions to fill, possible AAs)
        pass
    def apply_action(self, state, action):
        # Return new state after applying action
        pass
    def mask_low_plddt_positions(self, sequence, plddt_scores, num_to_mask):
        """
        Given a sequence and its plDDT scores, return a mask for the positions with lowest plDDT.
        Args:
            sequence: list or str of amino acids
            plddt_scores: list of floats, same length as sequence
            num_to_mask: int, number of positions to mask
        Returns:
            mask: list of bools or indices to mask
        """
        pass 
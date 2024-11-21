from pydantic import BaseModel, Field

class Advisor(BaseModel):
    name: str
    strategy: str  # e.g., 'Aggressive', 'Conservative', 'Opportunistic'

def generate_advisor_prompt_template():
    """
    Generates a formatted prompt string for an advisor in a Monopoly game, detailing
    the game's current state and the proposed action for evaluation.

    Returns:
        str: A prompt template string with placeholders for:
            - {advisor_name}: The name of the advisor.
            - {advisor_strategy}: The strategy used by the advisor.
            - {initial_bank}: Initial bank details.
            - {initial_board}: Initial board configuration.
            - {initial_roads}: List of roads.
            - {initial_properties}: List of properties.
            - {proposed_action}: The proposed action made by the leader.
    """

    return """
        You are {advisor_name}, an advisor with a {advisor_strategy} strategy in a Monopoly game. 
        Your role is to evaluate the leader's proposed action and determine whether to approve or reject it.

        Proposed Action:
        {proposed_action}

        Here is the current game state:

        Bank:
        {initial_bank}

        Board:
        {initial_board}

        Roads:
        {initial_roads}

        Properties:
        {initial_properties}

        Your Objective:
        Given the current state of the game and your strategy, evaluate the proposed action and decide 
        if it aligns with your strategic goals.

        Guidelines:
        1. Analyze each component of the game state to understand the implications of the proposed action.
        2. Consider immediate risks, potential opportunities, and the long-term impact of the action.
        3. Evaluate the alignment of the action with your {advisor_strategy} strategy.

        Instructions:
        - Provide your reasoning step-by-step to justify your decision.
        - Clearly state your final decision: approve or reject the proposed action.
    """

def parse_advisor_response(response):
    decision = response.decision.lower()
    if "approve" in decision or "accept" in decision or "yes" in decision:
        return True
    else:
        return False
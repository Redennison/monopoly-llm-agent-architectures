import json
import os
from dotenv import load_dotenv

from monosim.player import Player
from monosim.board import get_board, get_roads, get_properties, get_community_chest_cards, get_bank
from langchain.callbacks import get_openai_callback

# LangChain
from langchain_openai import ChatOpenAI
from langchain import OpenAI, LLMChain, PromptTemplate

# Advisors
from advisors import Advisor, parse_advisor_response, generate_advisor_prompt_template

# other imports
from pydantic import BaseModel, Field
from typing import List

# Load dotenv
load_dotenv()

"""### Game Functions
Wrappers that retrieves relevant game state(s)
"""

def initialize_game() -> dict:
    """
    Initializes a game with two players and sets up the bank, board, roads, properties,
    and community chest cards.

    Returns:
        dict: A dictionary containing the following:
            - "bank": Game's bank object.
            - "board": Main game board.
            - "roads": List of road objects.
            - "properties": List of property objects.
            - "community_chest_cards": Dictionary of community chest cards.
            - "players": List of two Player objects, with Player 1 first.
    """

    bank = get_bank()
    board = get_board()
    roads = get_roads()
    properties = get_properties()
    community_chest_cards = get_community_chest_cards()
    community_cards_deck = list(community_chest_cards.keys())

    player1 = Player('player1', 1, bank, board, roads, properties, community_cards_deck)
    player2 = Player('player2', 2, bank, board, roads, properties, community_cards_deck)

    player1.meet_other_players([player2])
    player2.meet_other_players([player1])

    return {
        "bank": bank,
        "board": board,
        "roads": roads,
        "properties": properties,
        "community_chest_cards": community_chest_cards,
        "players": [player1, player2] # For now, player 1 always comes first
    }

def get_current_state(players) -> dict:
    """
    Retrieves the current state of each player, including position, owned roads,
    money, mortgaged properties, and other status details.

    Args:
        players (list[Player]): List of Player objects in the game.

    Returns:
        dict: A dictionary containing:
            - "players": A list of dictionaries, each with a player's state.
    """

    current_state = {
        "players": [{"state": player.get_state()} for player in players]
    }
    return current_state

# Example usage of the above function
initial_state = initialize_game()
initial_state["bank"]

"""### Prompt Template

The following defines a customizable prompt template for an agent in a Monopoly game. Each part of the template is easily customizable using placeholders for various game elements.
"""

# The agent plays as Player 1 by default
agent_role = "Player 1"

# prompt template wrapper / hook, a function that returns a string
def prompt_template():
    """
    Generates a formatted prompt string for an agent in a Monopoly game, detailing
    the game's current state and guiding strategic decision-making.

    Returns:
        str: A prompt template string with placeholders for:
            - {agent_role}: The role of the agent in the game.
            - {initial_bank}: Initial bank details.
            - {initial_board}: Initial board configuration.
            - {initial_roads}: List of roads.
            - {initial_properties}: List of properties.

    Usage:
        Substitute placeholders to customize the prompt with the game state.
    """

    return  """
        You are the {agent_role} in a Monopoly game. Here is the current game state:

        Bank:
        {initial_bank}

        Board:
        {initial_board}

        Roads:
        {initial_roads}

        Properties:
        {initial_properties}

        Players:
        Player 1 and Player 2

        Your Objective:
        Given the current state of the game, make strategic moves that maximizes your chances of winning.

        Guidelines:
        1. Analyze each component of the game state to understand your current situation.
        2. Consider any immediate risks or opportunities from property ownership, player positions, or your current balance.

        Instructions:
        - Reason step-by-step to ensure your action aligns with the game’s rules and overall strategy.
        - Provide your next move by determining if you should buy the property or not (yes or no)
  """

"""**Sample Usage:**
```python
# Define the game setup and get initial game state
game = initialize_game()  # Initializes the bank, board, roads, properties, and players

# Generate the prompt with specific game details
template = prompt_template()
formatted_prompt = template.format(
    agent_role="Player 1",
    initial_bank=game["bank"],
    initial_board=game["board"],
    initial_roads=game["roads"],
    initial_properties=game["properties"]
)

print(formatted_prompt)
```

### Output Parser

The following defines a parser for interpreting the agent's output in the Monopoly game
"""

class Output(BaseModel):
    reasoning: str = Field(description="Your reasoning for the decision")
    decision: str = Field(description="Your decision for the next move")

def output_parser(model):
    return model.with_structured_output(Output)

"""### Simulate the Game

Set up prompt template and LLM chain
"""

model = ChatOpenAI(model="gpt-4o")

structured_llm = model.with_structured_output(Output)

"""Initialize the game and make arbitrary moves"""

game = initialize_game()

player1 = game["players"][0]
player2 = game["players"][1]
list_players = [player1, player2]

stop_at_round = 5 # arbitrary number of rounds to play before agent comes in and make a decision (for POC)

idx_count = 0
while not player1.has_lost() and not player2.has_lost() and idx_count < stop_at_round:
    for player in list_players:
        player.play()
    idx_count += 1

"""injecting variables

1. Set up prompt template and LLM chain
2. Hardcode some injection variables & make sure it works
3. Code to retrieve game info / states
"""

initial_template = prompt_template()

### Only one turn of the game is played so far
context = initial_template.format(
    agent_role="Player 1",  # or as appropriate
    initial_bank=game["bank"],
    initial_board=game["board"],
    initial_roads=game["roads"],
    initial_properties=game["properties"]
)

# Use the callback to measure tokens
with get_openai_callback() as cb:
    proposal = structured_llm.invoke(f"${context}. player_state is ${get_current_state(list_players)}")

    proposal_reasoning = proposal.reasoning 
    proposal_decision = proposal.decision

    # Display player's decision
    print("Player proposal:")
    print(proposal_decision)
    print("Player reasoning:")
    print(proposal_reasoning)

    print("Turning to advisors for a vote...")

    # Create advisors
    advisors = [
        Advisor(name='Advisor A', strategy='Aggressive'),
        Advisor(name='Advisor B', strategy='Conservative'),
        Advisor(name='Advisor C', strategy='Opportunistic')
    ]

    votes = []
    for advisor in advisors:
        # Prepare prompt for the advisor's LLM
        advisor_prompt = generate_advisor_prompt_template()

        # Add context to prompt template
        advisor_context = advisor_prompt.format(
            advisor_name=advisor.name,
            advisor_strategy=advisor.strategy,
            initial_bank=game["bank"],
            initial_board=game["board"],
            initial_roads=game["roads"],
            initial_properties=game["properties"],
            proposed_action=proposal_decision
        )
        
        # Use the LLM to decide whether to approve or reject the proposal
        advisor_response = structured_llm.invoke(advisor_context)
        
        # Parse the response to get the vote (True for approve, False for reject)
        vote = parse_advisor_response(advisor_response)
        votes.append(vote)
        print(f"{advisor.name} ({advisor.strategy}) votes to {'approve' if vote else 'reject'} the proposal.")

    # Determine if the proposal is accepted
    if votes.count(True) > votes.count(False):
        print("Proposal accepted by advisors.")
    else:
        print("Proposal rejected by advisors.")

    print(f"Total Tokens Used: {cb.total_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

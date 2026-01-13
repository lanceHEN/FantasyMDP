from typing import Dict, Tuple, List, Set
from collections import Counter

import pandas as pd
import numpy as np
from scipy.stats import geom

from utils.constants import POS_LIMITS, POS, NUM_POS

class FantasyDraftEnv:
    """Environment for the fantasy football draft.
    
    Observations are 1d numpy arrays of the form [num_qb_drafted, num_rb_drafted, num_wr_drafted, num_te_drafted, num_dst_drafted,
    num_k_drafted, best_qb_available_points, best_rb_available_points, best_wr_available_points,
    best_te_available_points, best_dst_available_points, best_k_available_points, round].
    Note the counts for drafted players are each normalized by dividing by the max allowed. Actions involve
    drafting the best available player for the given position. Rewards are projected total points.
    
    We simulate opponent choices by sampling from a geometric distribution among
    top ranked available players (we track what positions they can draft too).
    
    Attributes:
        player_df (pd.DataFrame): DataFrame containing ranked players with their position and projected points.
        pos_limits (Dict[str, int]): Mapping from each position to the max number of players to draft.
        num_teams (int): Number of teams in the draft.
        total_rounds (int): Number of rounds in the draft.
        first_pick_num (int): First pick number for the agent.
        geom_success (float): Probability of success in each trial for geometric distribution.
        draft_board (pd.DataFrame): DataFrame containing best available players; changes as draft progresses.
        team_selections (List[Dict[str, int]]): List of position count mappings, where the jth element contains
            drafted position counts for the jth team.
        total_rewards (float): Cumulative rewards found thus far.
        """
    
    def __init__(self, player_df: pd.DataFrame, first_pick_num: int, pos_limits: Dict[str, int] = POS_LIMITS,
                 num_teams: int = 12, geom_success: float=0.7):
        """
        Initializes a FantasyDraftEnv. NOTE the environment is not reset until
        reset() is explicitly called, since we need to return the initial observation.
        
        Args:
            player_df (pd.DataFrame): DataFrame containing ranked players with their position and projected points.
            first_pick_num (int): First pick number for the agent.
            pos_limits (Dict[str, int]): Mapping from each position to the max number of players to draft.
            num_teams (int): Number of teams in the draft.
            geom_success (float): Probability of success in each trial for geometric distribution.
        """
        
        self.player_df = player_df
        self.pos_limits = pos_limits
        self.num_teams = num_teams
        self.total_rounds = sum(pos_limits.values())
        self.first_pick_num = first_pick_num
        self.geom_success=geom_success
        
        self.draft_board = None
        self.team_selections = None
        self.round = None
        self.geom_success = None
        self.total_rewards = None
        
    # reset so that all players are available
    # we let geom_success represent the probability of success for the geometric distribution simulating other player selections
    # lower it for more random selections and raise it for more conservative, "chalky" selections.
    def reset(self) -> np.ndarray:
        """
        Resets the FantasyDraftEnv, starting the draft over and simulating any
        opponent picks prior to the agent's first pick, returning the initial
        observation tuple.
        
        Returns:
            np.ndarray: Observation of the form [num_qb_drafted, num_rb_drafted,
                num_wr_drafted, num_te_drafted, num_dst_drafted, num_k_drafted,
                best_qb_available_points, best_rb_available_points, best_wr_available_points,
                best_te_available_points, best_dst_available_points, best_k_available_points,
                round].
        """
        self.draft_board = self.player_df.copy()
        self.round = 1
        # positions drafted per team, to ensure others also abide by positional limit
        self.team_selections = [Counter() for _ in range(self.num_teams)]
        self.total_rewards = 0
        
        # play round till first player pick
        other_pick_teams = list(range(1, self.first_pick_num))
        self._simulate_other_picks(other_pick_teams)
        #print(self.draft_board)
        # get the current state
        return self._get_state()
        
    def step(self, action: str) -> Tuple[np.ndarray, float, bool, str]:
        """
        Drafts player at the given position to the agent's team, simulating
        any opponent picks thereafter by sampling from a geometric distribution.
        
        Args:
            action (str): What position to pick.
        
        Returns:
            Tuple[np.ndarray, float, bool, str]: The new observation array,
                reward, whether the game ended (were all rounds played), and
                name of player drafted. Observation arrays have the form [num_qb_drafted, num_rb_drafted,
                num_wr_drafted, num_te_drafted, num_dst_drafted, num_k_drafted,
                best_qb_available_points, best_rb_available_points, best_wr_available_points,
                best_te_available_points, best_dst_available_points, best_k_available_points,
                round]. Note the counts for drafted players are each normalized by
                dividing by the max allowed.
        """
        
        # find best player for the given action
        max_idx = self._get_max_idx(action)
        
        # simulate drafting player
        player_name, reward = self._draft_player(max_idx, self.first_pick_num)
        
        self.total_rewards += reward
        
        # now, we must simulate the process for all the other teams, up until the next player selection.
        # that means finishing up this round as needed, then playing the next round until the next selection unless done.
            
        self.round += 1
        done = self.round > self.total_rounds
        
        if not done: # if done, don't need to simulate any more picks
            # get list of one-indexed team indices in order of draft pick, until the next player selection
            if self.round % 2 != 0: # odd round (first, third, fifth, ...)
                other_pick_teams = list(range(self.first_pick_num + 1, self.num_teams + 1)) + list(range(self.num_teams, self.first_pick_num, -1))  
            else: # even round (second, fourth, sixth, ...)
                other_pick_teams = list(range(self.first_pick_num - 1, 1 - 1, -1)) + list(range(1, self.first_pick_num))
                
            self._simulate_other_picks(other_pick_teams)
        
        new_state = self._get_state()
        
        return new_state, self.total_rewards if done else 0, done, player_name
    

    def _simulate_other_picks(self, other_pick_teams: List[int]) -> None:
        """
        Simulates selections for each team given by their 0-based index from
        other_pick_teams.
        
        Args:
            other_pick_teams (List[int]): Ordered list of teams to draft players
                for, where the first team drafts first. The team is represented
                by its 1-based index, so the team that picked first in the first
                round has index 1 and the team that picked last has index
                self.num_teams.
        """
        for other_pick_team in other_pick_teams:
            available_pos = self._get_available_pos(other_pick_team)
            
            filtered = self.draft_board[self.draft_board['POS'].isin(available_pos)]
            #print(filtered)
            # sample from geom dist
            to_draft = geom.rvs(self.geom_success) - 1 # subtract one to be zero-based
            
            if to_draft >= len(filtered):
                to_draft = len(filtered) - 1  # clip to valid index if needed

            # get the index label of the to_draft-th row
            original_index = filtered.index[to_draft]

            # simulate drafting player
            self._draft_player(original_index, other_pick_team)
            
    def _get_max_idx(self, pos: str):
        """
        Finds the index for the row in self.draft_board containing the highest
        projected points for the given position.
        
        Args:
            pos (str): The position to find the highest projectwed points for.
        """
        filtered = self.draft_board[(self.draft_board['POS'] == pos)]
        max_idx = filtered['POINTS'].idxmax()
        
        return max_idx
    
    def _draft_player(self, player_index: int, team_idx: int) -> Tuple[str, float]:
        """
        Drafts the player at the given index in self.draft_board to the given team.
        
        The player will be removed from self.draft_board, and the team's
        positional counts will be updated.
        
        Args:
            player_index (int): 0-based index of the player in self.draft_board.
            team_idx (int): 1-based index of the team, so the team that picked
                first in the first round has index 1 and the team that picked
                last has index self.num_teams.
                
        Returns:
            Tuple[str, float]: The name and projected points of the player.
        """
        row = self.draft_board.loc[player_index]
        #print(self.draft_board.head(20))
        #print(player_index)
        #print(row)
        
        # get points and player name
        points = row['POINTS']
        player_name = row['PLAYER NAME']
        
        # add position to selections
        self.team_selections[team_idx - 1][row['POS']] += 1 # Counter so indexing always works
        
        # remove row from draft board
        self.draft_board = self.draft_board.drop(player_index)
        
        return player_name, points        
            
    # generate the state tuple corresponding with the current internal state
    def _get_state(self) -> np.ndarray:
        """
        Produces an observational array for the current state of the draft.
        
        Returns:
            np.ndarray: Observation of the form [num_qb_drafted, num_rb_drafted,
                num_wr_drafted, num_te_drafted, num_dst_drafted, num_k_drafted,
                best_qb_available_points, best_rb_available_points, best_wr_available_points,
                best_te_available_points, best_dst_available_points, best_k_available_points,
                round]. Note the counts for drafted players are each normalized
                by dividing by the max allowed.
        """
        
        team_selections = self.team_selections[self.first_pick_num - 1]
        
        # get per_position counts
        pos_counts = []
        
        top_points = []
        # find positional counts, top points per position (each normalized)
        for pos in POS:
            pos_counts.append(team_selections.count(pos) / self.pos_limits[pos]) # normalize
            
            max_idx = self._get_max_idx(pos)
            row = self.draft_board.loc[max_idx]
            top_points.append(row['POINTS'])
            
        return np.array([pos_counts] + [top_points] + [self.round])
        
    def _get_available_pos(self, team_num: int) -> Set[str]:
        """
        Produces set of available position names, for the given team (one indexed).
        """
        team_selections = self.team_selections[team_num - 1]
        
        available = set()
        for pos in POS:
            if team_selections.count(pos) < POS_LIMITS[pos]:
                available.add(pos)
                
        #print(team_selections)
        #print(available)
                
        return available
    
    def _get_pick(self, round: int) -> int:
        """
        Calculates the agent's overall pick number for the given one-indexed round.
        """
        if round % 2 != 0: # odd (first, third, fifth, ... rounds)
            return self.first_pick_num + (round-1) * self.num_teams
        else: # even (second, fourth, sixth, ... rounds)
            return (self.num_teams - self.first_pick_num + 1) + (round-1) * self.num_teams
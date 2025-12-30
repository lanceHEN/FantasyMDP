from utils.constants import POS_LIMITS

class FantasyDraftEnv:
    """Environment for the fantasy football draft. Observations are tuples of the form (num_qb_drafted (normalized),
    num_rb_drafted (normalized), num_wr_drafted (normalized), num_te_drafted (normalized), num_dst_drafted (normalized),
    num_k_drafted (normalized), best_qb_available_points, best_rb_available_points, best_wr_available_points,
    best_te_available_points, best_dst_available_points, best_k_available_points, round (normalized)). Actions involve
    drafting the best available player for the given position."""
    
    # initialize with the given player dataframe, storing the players with their positions, projected points, and tiers
    # as well as with the first pick number (one based), positional limits, and number of teams.
    def __init__(self, player_df, first_pick_num, pos_limits = POS_LIMITS, num_teams = 12):
        self.player_df = player_df
        self.pos_limits = pos_limits
        self.num_teams = num_teams
        self.total_rounds = sum(pos_limits.values())
        self.first_pick_num = first_pick_num
        
        self.draft_board = None
        # positions drafted per team, to ensure others also abide by positional limit
        self.team_selections = None
        self.round = None
        self.geom_success = None
        self.total_rewards = None
        self.zero_pos_fractions = None
        
    # reset so that all players are available
    # we let geom_success represent the probability of success for the geometric distribution simulating other player selections
    # lower it for more random selections and raise it for more conservative, "chalky" selections.
    def reset(self, geom_success=0.4, zero_pos_fractions=False):
        self.draft_board = self.player_df.copy()
        self.round = 1
        self.geom_success = geom_success
        # positions drafted per team, to ensure others also abide by positional limit
        self.team_selections = [[] for _ in range(self.num_teams)]
        self.total_rewards = 0
        self.zero_pos_fractions = zero_pos_fractions
        
        # play round till first player pick
        other_pick_teams = list(range(1, self.first_pick_num))
        self._simulate_other_picks(other_pick_teams)
        #print(self.draft_board)
        # get the current state
        return self._get_state()
        
        
    # Draft a player to the team, returning the new state tuple, reward, whether the game ended (were all rounds played), and who was drafted.
    # To find the next state, we have to simulate other players making moves - we use a geometric distribution
    # over the ranked players available at their position, setting p=0.3 simply because it sounds good enough
    def step(self, action, geom_success=0.7):
        
        # find best player for the given action
        max_idx = self._get_max_idx(action)
        
        # simulate drafting player
        player_name, reward = self._draft_player(max_idx, self.first_pick_num)
        
        self.total_rewards += reward
        
        # now, we must simulate the process for all the other teams, up until the next player selection.
        # that means finishing up this round as needed, then playing the next round until the next selection.
        
        # get list of one-indexed team indices in order of draft pick, until the next player selection
        if self.round % 2 != 0: # odd round (first, third, fifth, ...)
            other_pick_teams = list(range(self.first_pick_num + 1, self.num_teams + 1)) + list(range(self.num_teams, self.first_pick_num, -1))
            
        else: # even round (second, fourth, sixth, ...)
            other_pick_teams = list(range(self.first_pick_num - 1, 1 - 1, -1)) + list(range(1, self.first_pick_num))
            
        self.round += 1
        done = self.round > self.total_rounds
        
        if not done: # if done, don't need to simulate any more picks
            self._simulate_other_picks(other_pick_teams)
        
        new_state = self._get_state()
        
        return new_state, self.total_rewards if done else 0, done, player_name
    
    # simulate other team draft selectoins, where they can also only draft players at positions that don't exceed positional limits.
    # we simulate the stochasticity by sampling from a geometric dist. over the ranked available players
    def _simulate_other_picks(self, other_pick_teams):
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
            
    # find the index for the row in self.draft_board containing the highest projected points for the given position
    def _get_max_idx(self, pos):
        filtered = self.draft_board[(self.draft_board['POS'] == pos)]
        max_idx = filtered['POINTS'].idxmax()
        
        return max_idx
    
    # simulates drafting the given player, for the given team.
    # removes the player from the draft board, adds their position to the list of positions drafted for that team,
    # and returns their name and projected points
    def _draft_player(self, player_index, team_num):
        row = self.draft_board.loc[player_index]
        #print(self.draft_board.head(20))
        #print(player_index)
        #print(row)
        
        # get points and player name
        points = row['POINTS']
        player_name = row['PLAYER NAME']
        
        # add position to selections
        self.team_selections[team_num - 1].append(row['POS'])
        
        # remove row from draft board
        self.draft_board = self.draft_board.drop(player_index)
        
        return player_name, points        
            
    # generate the state tuple corresponding with the current internal state
    def _get_state(self):
        
        team_selections = self.team_selections[self.first_pick_num - 1]
        
        # get per_position counts
        pos_counts = []
        
        top_points = []
        # find positional counts, top points per position (each normalized)
        for pos in POS:
            if self.zero_pos_fractions:
                pos_counts.append(0)
            else:
                pos_counts.append(team_selections.count(pos) if POS_LIMITS[pos] == 1 else team_selections.count(pos) / POS_LIMITS[pos]) # normalize
            
            max_idx = self._get_max_idx(pos)
            row = self.draft_board.loc[max_idx]
            top_points.append(row['POINTS'] - POS_BASELINE_POINTS[pos])
            
        return tuple(pos_counts) + tuple(top_points) + (0,) #(self.round,)
        
    # get set of available positions, for the given team (one indexed)
    def _get_available_pos(self, team_num):
        team_selections = self.team_selections[team_num - 1]
        
        available = set()
        for pos in POS:
            if team_selections.count(pos) < POS_LIMITS[pos]:
                available.add(pos)
                
        #print(team_selections)
        #print(available)
                
        return available
    
    # calculate the pick, given the round
    # assumes snaking order
    # round is one-indexed
    def _get_pick(self, round):
        if round % 2 != 0: # odd (first, third, fifth, ... rounds)
            return self.first_pick_num + (round-1) * self.num_teams
        else: # even (second, fourth, sixth, ... rounds)
            return (self.num_teams - self.first_pick_num + 1) + (round-1) * self.num_teams
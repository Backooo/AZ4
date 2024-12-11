import numpy as np

class MCTSNode:
    def __init__(self, parent, play, state, unexpanded_plays):
        self.play = play
        self.state = state
        self.n_plays = 0
        self.n_wins = 0
        self.parent = parent
        self.children = {
            hash(play): {"play": play, "node": None} for play in unexpanded_plays
        }

    def is_fully_expanded(self):
        return all(child["node"] is not None for child in self.children.values())

    def best_child(self, exploration_param=1.41):
        def ucb1(child_node):
            if child_node.n_plays == 0:
                return float('inf')  # Prioritize unexplored nodes
            return (child_node.n_wins / child_node.n_plays) + \
                   exploration_param * np.sqrt(np.log(self.n_plays) / child_node.n_plays)

        children_nodes = [child["node"] for child in self.children.values() if child["node"] is not None]
        if not children_nodes:
            return None
        return max(children_nodes, key=ucb1)

    def expand(self, play, state, unexpanded_plays):
        child_node = MCTSNode(self, play, state, unexpanded_plays)
        self.children[hash(play)]["node"] = child_node
        return child_node

    def backpropagate(self, result):
        self.n_plays += 1
        if result:
            self.n_wins += 1
        if self.parent:
            self.parent.backpropagate(result)
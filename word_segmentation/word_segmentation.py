import lm

class SegmentationProblem(object):
    def __init__(self, query, unigram_cost):
        self.query = query
        self.unigram_cost = unigram_cost

    def start_state(self):
        return 0

    def is_end(self, state):
        return state == len(self.query)

    def subproblems(self, state):
        results = []
        for l in range(1, len(self.query) - state + 1):
            target_word = self.query[state:state+l]
            next_state = state + l
            cost = self.unigram_cost(target_word) 
            results.append((target_word, next_state, cost))
        return results

class DynamicProgrammingSearch(object):
    def __init__(self, memory_use=True, verbose=0):
        self.memory_use = memory_use
        if memory_use:
            self.subproblem_cache = {}
        self.verbose = verbose

    def suboptimal(self, state):
        if self.memory_use and state in self.subproblem_cache:
            min_actions, min_cost, num_visited = self.subproblem_cache[state]
            return min_actions, min_cost, 1

        # check whether it's in the last state
        num_visited = 1
        if self.problem.is_end(state):
            # if the last state return 0 cost
            min_actions = []
            min_cost = 0
        else:
            # otherwise calculate all possible sub problems
            # and return minimum (sub obtimal) cost
            min_cost = float('inf')
            min_actions = []
            sub_problems = self.problem.subproblems(state)
            for action, new_state, cost in sub_problems:
                f_actions, f_cost, f_num_visited = self.suboptimal(new_state)
                num_visited += f_num_visited
                if f_cost + cost < min_cost:
                    min_cost = f_cost + cost
                    min_actions = [action] + f_actions

        if self.memory_use:
            self.subproblem_cache[state] = min_actions, min_cost, num_visited

        return min_actions, min_cost, num_visited

    def solve(self, problem):
        self.problem = problem
        actions, cost, num_visited = self.suboptimal(problem.start_state())
        if self.verbose >= 1:
            print("num states visited = {}".format(num_visited))
            print("total cost = {}".format(cost))
            print("actions = {}".format(actions))
        return actions, cost, num_visited

unigram_cost, bigram_cost = lm.make_LM('war_and_peace.txt')
problem = SegmentationProblem('whatdoesthisreferto', unigram_cost)

dps = DynamicProgrammingSearch(memory_use=True, verbose=1)
dps.solve(problem)

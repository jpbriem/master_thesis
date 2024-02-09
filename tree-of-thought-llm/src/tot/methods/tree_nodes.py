import copy

class Node:
    nodeID = 0
    nodes = []
    def __init__(self, task_name, level, x, phase="abstraction", LLM_answer="", thought="", value=0, parent=None, n_generate_children=1, children=[], leaf=False):
        self.nodeID = Node.nodeID
        Node.nodeID += 1
        self.task_name = task_name
        self.x = x
        self.level = level # level in the tree
        self.phase = phase # phase: Abstraction of a Pattern vs. Application of pattern to test case
        if level == 0:
            self.isRoot = True
        else:
            self.isRoot = False
        self.isLeaf = leaf
        self.n_generate_children = n_generate_children
        self.children = children
        self.parent = parent
        self.LLM_answer = LLM_answer 
        self.thought = thought # Extract the thought from the LLM answer
        self.thought_before_revision = thought
        self.value = value # valuation of thought 
        self.current_test_idx = None # index of example currently under test 
        self.example_success = None # list of booleans indicating whether the thought is correct on the examples
        self.revisions_total = None # total number of revisions needed to find a correct abstraction
        Node.nodes.append(self)

    def __repr__(self) -> str:
        return f"{self.task_name}-Node_{self.nodeID}(Level: {self.level}, Phase: {self.phase}, Thought: {self.thought}, Value: {self.value}, Parent_ID: {self.parent.nodeID if not self.isRoot else None}, Spread: {True if self.n_generate_children>1 else False}, Children_ID: {[child.nodeID for child in self.children]}, is_root: {self.isRoot}, is_leaf: {self.isLeaf})"
    
    def copy(self):
        copied_children = copy.deepcopy(self.children)
        return Node(self.task_name, self.level, self.x, self.phase, self.LLM_answer, self.thought, self.value, self.parent, self.n_generate_children, copied_children, self.isLeaf)
    
    def reset_tree():
        Node.nodeID = 0
        Node.nodes = []
        
        
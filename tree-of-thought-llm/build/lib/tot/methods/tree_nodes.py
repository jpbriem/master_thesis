class Node:
    nodeID = 0
    nodes = []
    def __init__(self, level, x, phase="abstraction", LLM_answer="", thought="", value=0, parent=None, n_generate_children=1, children=[], leaf=False):
        self.nodeID = Node.nodeID
        Node.nodeID += 1
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
        self.value = value # valuation of thought
        Node.nodes.append(self)

    def __repr__(self) -> str:
        return f"Node_{self.nodeID}(Level: {self.level}, Phase: {self.phase}, Thought: {self.thought}, Value: {self.value}, Parent_ID: {self.parent.nodeID if not self.isRoot else None}, Spread: {True if self.n_generate_children>1 else False}, Children_ID: {[child.nodeID for child in self.children]}, is_root: {self.isRoot}, is_leaf: {self.isLeaf})"
    
    
    def reset_tree():
        Node.nodeID = 0
        Node.nodes = []
        
        
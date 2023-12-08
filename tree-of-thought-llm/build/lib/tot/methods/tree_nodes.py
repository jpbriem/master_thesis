class Node:
    nodeID = 0
    nodes = []
    def __init__(self, level, x, LLM_answer="", thought="", value=0, parent=None, children=[], leaf=False):
        self.nodeID = Node.nodeID
        Node.nodeID += 1
        self.level = level # level in the tree
        if level == 0:
            self.isRoot = True
        else:
            self.isRoot = False
        self.LLM_answer = LLM_answer # TODO: seperate between whole answer and partial answer
        self.thought = thought
        self.value = value
        self.children = children
        self.parent = parent
        self.isLeaf = leaf
        self.x = x
        Node.nodes.append(self)

    def __repr__(self) -> str:
        return f"Node_{self.nodeID}(Level: {self.level}, Thought: {self.thought}, LLM answer: {self.LLM_answer}, Value: {self.value}, Parent_ID: {self.parent.nodeID if not self.isRoot else None}, Children_ID: {[child.nodeID for child in self.children]}, is_root: {self.isRoot}, is_leaf: {self.isLeaf})"
    
    
    def reset_tree():
        Node.nodeID = 0
        Node.nodes = []
        
        
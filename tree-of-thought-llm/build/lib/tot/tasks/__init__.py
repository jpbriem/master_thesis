def get_task(name):
    if name == 'game24':
        from tot.tasks.game24 import Game24Task
        return Game24Task()
    elif name == 'text':
        from tot.tasks.text import TextTask
        return TextTask()
    elif name == 'crosswords':
        from tot.tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask()
    elif name == 'arc':
        from tot.tasks.arc_full_plan import ARCTask
        return ARCTask()
    elif name == "arc_1D":
        from tot.tasks.arc_1D import ARC_1D
        return ARC_1D()
    elif name == "arc_h_v":
        from tot.tasks.arc_h_v import ARC_h_v
        return ARC_h_v()
    else:
        raise NotImplementedError
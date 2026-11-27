"""Independent head conversion with preserved AdamW moments and EMA state."""
import copy


def apply_head_split(student, teacher, optimizer):
    if student.head_layout == teacher.head_layout == 'separate':
        return False
    if student.head_layout != 'shared' or teacher.head_layout != 'shared':
        raise ValueError('head split requires two shared heads')
    if len(optimizer.param_groups) != 1:
        raise ValueError('head split requires one AdamW group')
    params = optimizer.param_groups[0]['params']
    ids = {id(p) for p in params}
    # Validate every source before mutating either model or optimizer.
    states = []
    for parameter in student.dino_head.parameters():
        state = optimizer.state.get(parameter, {})
        if id(parameter) not in ids or not {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys():
            raise ValueError('head split requires complete AdamW moments')
        states.append(copy.deepcopy(state))
    student.ibot_head = copy.deepcopy(student.dino_head)
    teacher.ibot_head = copy.deepcopy(teacher.dino_head)
    for parameter, state in zip(student.ibot_head.parameters(), states):
        params.append(parameter)
        optimizer.state[parameter] = state
    student.head_layout = teacher.head_layout = 'separate'
    return True

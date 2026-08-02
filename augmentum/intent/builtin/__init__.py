"""Builtin actions — control, navigation, start, work.

Each module registers its actions on import; ``augmentum.intent.__init__``
eager-imports them so the decorators run at process start.
"""

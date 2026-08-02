"""Cast — render pipeline + TV-target coordination.

Houses the render-side foundation types (RenderJob / RenderRoute /
kind constants) that the router director, the actual render pipeline
(future), and the TV receiver protocol (future) all consume.

Deliberately kept independent of augmentum.fabric so single-machine
deployments use these types without pulling fabric machinery.
"""

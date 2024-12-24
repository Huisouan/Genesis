import genesis as gs
import numpy 
gs.init(backend=gs.cpu)
base_init_pos = [0.0, 0.0, 0.42]
base_init_quat = [0.0, 0.0, 0.0, 1.0]
scene = gs.Scene(show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/go2/urdf/go2.urdf",
        pos=base_init_pos,
        quat=base_init_quat,
    ),
)

scene.build()

for i in range(1000):
    scene.step()
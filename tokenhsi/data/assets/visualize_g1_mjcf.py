"""Visualize mjcf/g1.xml (MuJoCo MJCF) in viser with interactive joint sliders.

g1.xml's meshdir ("../meshes/g1") isn't present in this repo, so meshes are
sourced from the downloaded URDF assets at assets/g1_with_brainco_hand/meshes
instead (same mesh filenames), by rewriting meshdir before loading.

Usage:
    python visualize_g1_mjcf.py
"""

import re
import time
from pathlib import Path

import mujoco
import trimesh
import viser
import viser.transforms as vtf

MJCF_PATH = Path(__file__).parent / "mjcf" / "g1.xml"
MESH_DIR = Path(__file__).resolve().parents[3] / "assets" / "g1_with_brainco_hand" / "meshes"


def load_model() -> mujoco.MjModel:
    xml_text = MJCF_PATH.read_text()
    xml_text = re.sub(r'meshdir="[^"]*"', f'meshdir="{MESH_DIR}"', xml_text)
    return mujoco.MjModel.from_xml_string(xml_text)


def mesh_for_geom(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
    mesh_id = model.geom_dataid[geom_id]
    vert_start = model.mesh_vertadr[mesh_id]
    vert_count = model.mesh_vertnum[mesh_id]
    face_start = model.mesh_faceadr[mesh_id]
    face_count = model.mesh_facenum[mesh_id]
    vertices = model.mesh_vert[vert_start : vert_start + vert_count]
    faces = model.mesh_face[face_start : face_start + face_count]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def main():
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4, height=4)

    # One viser mesh handle per visual-group mesh geom (group 2, per g1.xml's "visual" class).
    mesh_handles = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_group[geom_id] != 2:
            continue
        mesh = mesh_for_geom(model, geom_id)
        handle = server.scene.add_mesh_trimesh(f"/geoms/geom_{geom_id}", mesh)
        mesh_handles.append((geom_id, handle))

    def sync_geom_transforms():
        for geom_id, handle in mesh_handles:
            handle.position = data.geom_xpos[geom_id]
            handle.wxyz = vtf.SO3.from_matrix(data.geom_xmat[geom_id].reshape(3, 3)).wxyz

    sync_geom_transforms()

    # Sliders for the actuated hinge joints (the pelvis freejoint stays at its qpos0 default).
    slider_handles = []
    joint_qpos_adrs = []
    with server.gui.add_folder("Joints"):
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = model.joint(joint_id).name
            lower, upper = model.jnt_range[joint_id]
            if lower == upper == 0.0:
                lower, upper = -3.14, 3.14
            slider = server.gui.add_slider(
                label=name, min=float(lower), max=float(upper), step=1e-3, initial_value=0.0
            )
            slider_handles.append(slider)
            joint_qpos_adrs.append(model.jnt_qposadr[joint_id])

    def update_config():
        for slider, qpos_adr in zip(slider_handles, joint_qpos_adrs):
            data.qpos[qpos_adr] = slider.value
        mujoco.mj_forward(model, data)
        sync_geom_transforms()

    for slider in slider_handles:
        slider.on_update(lambda _: update_config())

    with server.gui.add_folder("Controls"):
        reset_button = server.gui.add_button("Reset joints to 0")

        @reset_button.on_click
        def _(_):
            for slider in slider_handles:
                slider.value = 0.0
            update_config()

    print("Viser server running. Open the printed URL in your browser.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

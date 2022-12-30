import copy
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import pyrender
import quaternion
import trimesh


class Retargeting:
    @classmethod
    def array_to_tuple(cls, array):
        return tuple(
            cls.array_to_tuple(i) if isinstance(i, np.ndarray) else i for i in array
        )

    @classmethod
    def from_dir(cls, path):
        path = Path(path)
        mesh_path = path / path.with_suffix(".glb").name
        data_path = path / path.with_suffix(".npz").name

        data = np.load(data_path)
        C = data["positions"]
        BE = data["edges"]
        P = data["parents"]
        W = data["weight"]

        mesh = pyrender.Mesh.from_trimesh(
            list(trimesh.load(mesh_path).geometry.values())
        )
        V = np.concatenate([primitive.positions for primitive in mesh.primitives])
        return cls(C, BE, P, V, W, mesh)

    def __init__(self, C, BE, P, V, W, mesh, width=1920, height=1080) -> None:
        self.C = jnp.array(C, dtype=jnp.float32)
        self.BE = self.array_to_tuple(BE)
        self.P = self.array_to_tuple(P)

        self.V = jnp.array(V, dtype=jnp.float32)
        self.W = jnp.array(W, dtype=jnp.float32)

        self.bone_count = len(BE)

        self.dQ = jnp.array([[1, 0, 0, 0]], dtype=jnp.float32).repeat(
            repeats=self.bone_count, axis=0
        )
        self.dT = jnp.array([[0, 0, 0]], dtype=jnp.float32).repeat(
            repeats=self.bone_count, axis=0
        )

        self.vQ = jnp.empty_like(self.dQ)
        self.vD = jnp.empty_like(self.vQ)
        self.vT = jnp.empty_like(self.dT)
        self.CD = jnp.empty_like(self.C)
        self.CT = jnp.empty_like(self.C)

        self.mesh = mesh
        self.width = width
        self.height = height

        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.width, viewport_height=self.height, point_size=1.0
        )
        self.scene = pyrender.Scene()
        self.camera = pyrender.PerspectiveCamera(
            np.pi / 3, aspectRatio=self.width / self.height
        )
        self.light = pyrender.DirectionalLight(intensity=2)
        self.node_mesh = pyrender.Node(mesh=copy.deepcopy(self.mesh))

        self.scene.add_node(self.node_mesh)

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [0, 1.5, 3]
        self.scene.add(self.camera, pose=camera_pose)
        self.scene.add(self.light)

        self.forward_compiled = self.__forward_no_malloc.lower(
            self.dQ, self.dT, self.C, self.BE, self.P, self.vQ, self.vT, self.CD
        ).compile()

        self.dqs_compiled = self.__dqs.lower(
            self.vQ, self.vT, self.V, self.W, self.vD
        ).compile()

        self.retargeting_compiled = self.__retargeting_no_malloc.lower(
            self.C, self.BE, self.P, self.C, self.CT
        ).compile()

    def compile_inverse(self, optimizer=None, **kwargs):
        optimizer = optimizer or jaxopt.GradientDescent
        self.optimizer = optimizer(
            fun=partial(self.__loss_no_malloc, BE=self.BE, P=self.P),
            has_aux=True,
            **kwargs,
        )
        self.inverse_compiled = self.__inverse_no_malloc.lower(
            self.dQ, self.dT, self.C, self.vQ, self.vT, self.CD, self.CD, self.optimizer
        ).compile()

    def forward(self, dQ=None, dT=None, C=None):
        dQ = dQ if dQ is not None else self.dQ
        dT = dT if dT is not None else self.dT
        C = C if C is not None else self.C
        return self.forward_compiled(dQ, dT, C, self.vQ, self.vT, self.CD)

    def inverse(self, CT=None, dQ=None, dT=None, C=None):
        CT = CT if CT is not None else self.CT
        dQ = dQ if dQ is not None else self.dQ
        dT = dT if dT is not None else self.dT
        C = C if C is not None else self.C
        return self.inverse_compiled(dQ, dT, C, self.vQ, self.vT, self.CD, CT)

    def dqs(self, vQ=None, vT=None, V=None, W=None):
        vQ = vQ if vQ is not None else self.vQ
        vT = vT if vT is not None else self.vT
        V = V if V is not None else self.V
        W = W if W is not None else self.W
        return self.dqs_compiled(vQ, vT, V, W, self.vD)

    def retargeting(self, CR, C=None):
        C = C if C is not None else self.C
        return self.retargeting_compiled(C, CR, self.CT)

    def render(self, vQ=None, vT=None, V=None, W=None):
        U = self.dqs(vQ, vT, V, W)
        offset = 0
        new_mesh = copy.deepcopy(self.mesh)
        for primitive in new_mesh.primitives:
            num_pos = len(primitive.positions)
            primitive.positions = U[offset : offset + num_pos]
            offset += num_pos
        self.node_mesh.mesh = new_mesh
        return self.renderer.render(
            self.scene,
            pyrender.constants.RenderFlags.OFFSCREEN
            | pyrender.constants.RenderFlags.RGBA,
        )

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=(
            "BE",
            "P",
        ),
    )
    def __retargeting_no_malloc(C, BE, P, CR, CT):
        for i in range(len(BE)):
            p = P[i]
            i0 = BE[i][0]
            i1 = BE[i][1]
            c0 = C[i0]
            c1 = C[i1]
            cr0 = CR[i0]
            cr1 = CR[i1]
            v = (c1 - c0) / jnp.linalg.norm(c1 - c0)
            vr = (cr1 - cr0) / jnp.linalg.norm(cr1 - cr0)
            q = quaternion.from_norm_vector(v, vr)
            if p < 0:
                CT = CT.at[i0].set(cr0)
                CT = CT.at[i1].set(
                    quaternion.rotate(q, c1) + cr0 - quaternion.rotate(q, c0)
                )
            else:
                p1 = CT[BE[p][1]]
                CT = CT.at[i0].set(p1)
                CT = CT.at[i1].set(
                    quaternion.rotate(q, c1) + p1 - quaternion.rotate(q, c0)
                )
        return CT

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=(
            "BE",
            "P",
        ),
    )
    def __forward_no_malloc(dQ, dT, C, BE, P, vQ, vT, CD):
        for i in range(len(BE)):
            p = P[i]
            i0 = BE[i][0]
            i1 = BE[i][1]
            c0 = C[i0]
            c1 = C[i1]
            if p < 0:
                vQ = vQ.at[i].set(dQ[i])
                vT = vT.at[i].set(c0 - quaternion.rotate(dQ[i], c0) + dT[i])
            else:
                vQ = vQ.at[i].set(quaternion.multiply(vQ[p], dQ[i]))
                vT = vT.at[i].set(
                    vT[p] - quaternion.rotate(vQ[i], c0) + quaternion.rotate(vQ[p], c0)
                )
            CD = CD.at[i0].set(quaternion.rotate(vQ[i], c0) + vT[i])
            CD = CD.at[i1].set(quaternion.rotate(vQ[i], c1) + vT[i])
        return vQ, vT, CD

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=(
            "BE",
            "P",
        ),
    )
    def __loss_no_malloc(params, C, BE, P, vQ, vT, CD, CT):
        vQ, vT, CD = Retargeting.__forward_no_malloc(
            params["dQ"], params["dT"], C, BE, P, vQ, vT, CD
        )
        return jnp.sum((CD - CT) ** 2), (vQ, vT)

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=("optimizer",),
    )
    def __inverse_no_malloc(dQ, dT, C, vQ, vT, CD, CT, optimizer):
        params = {"dQ": dQ, "dT": dT}
        params, state = optimizer.run(params, C=C, vQ=vQ, vT=vT, CD=CD, CT=CT)
        return params, state

    @staticmethod
    @jax.jit
    def __dqs(vQ, vT, V, W, vD):
        vD = vD.at[..., 0].set(
            -0.5
            * (
                vT[..., 0] * vQ[..., 1]
                + vT[..., 1] * vQ[..., 2]
                + vT[..., 2] * vQ[..., 3]
            )
        )
        vD = vD.at[..., 1].set(
            0.5
            * (
                vT[..., 0] * vQ[..., 0]
                + vT[..., 1] * vQ[..., 3]
                - vT[..., 2] * vQ[..., 2]
            )
        )
        vD = vD.at[..., 2].set(
            0.5
            * (
                -vT[..., 0] * vQ[..., 3]
                + vT[..., 1] * vQ[..., 0]
                + vT[..., 2] * vQ[..., 1]
            )
        )
        vD = vD.at[..., 3].set(
            0.5
            * (
                vT[..., 0] * vQ[..., 2]
                - vT[..., 1] * vQ[..., 1]
                + vT[..., 2] * vQ[..., 0]
            )
        )
        b0 = W @ vQ
        be = W @ vD
        norm_b0 = jnp.linalg.norm(b0, ord=2, axis=-1)[..., None]
        c0 = b0 / norm_b0
        ce = be / norm_b0
        a0 = c0[..., 0, None]
        ae = ce[..., 0, None]
        d0 = c0[..., 1:]
        de = ce[..., 1:]
        return (
            V
            + 2 * jnp.cross(d0, (jnp.cross(d0, V) + a0 * V))
            + 2 * (a0 * de - ae * d0 + jnp.cross(d0, de))
        )

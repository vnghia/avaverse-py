import jax.numpy as jnp


def multiply(q1, q2):
    w1, x1, y1, z1 = jnp.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = jnp.split(q2, 4, axis=-1)
    ow = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ox = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    oy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    oz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.stack((ow, ox, oy, oz), -1).squeeze()


def rotate(q, v):
    s = q[..., 0, None]
    r = q[..., 1:]
    m = (
        q[..., 0, None] ** 2
        + q[..., 1, None] ** 2
        + q[..., 2, None] ** 2
        + q[..., 3, None] ** 2
    )
    return v + 2 * jnp.cross(r, (s * v + jnp.cross(r, v))) / m


def from_norm_vector(v1, v2):
    q = jnp.empty(4)
    q = q.at[1:].set(jnp.cross(v1, v2))
    q = q.at[0].set(1 + jnp.dot(v1, v2))
    return q / jnp.linalg.norm(q)

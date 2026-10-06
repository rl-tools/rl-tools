import numpy as np

from ._component import load_core

JOINT_SIZES = {"fixed": 0, "revolute": 1, "prismatic": 1, "spherical": 4}


def _identities(count):
    return np.broadcast_to(np.eye(4, dtype=np.float32)[:3], (count, 3, 4)).copy()


def _rows(value, count, what):
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.shape != (count, 3, 4):
        raise ValueError(f"{what} must have shape ({count}, 3, 4), got {array.shape}")
    return array


def _output(out, shape):
    if out is None:
        return np.empty(shape, dtype=np.float32)
    if (not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.shape != shape
            or not out.flags.c_contiguous or not out.flags.writeable):
        raise ValueError(f"out must be a writable, C-contiguous float32 array with shape {shape}")
    return out


def _homogeneous(transform):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3] = transform
    return matrix


class Rig:
    """Rigid-link tree over 3x4 [R|t] transforms: world_link = world_parent @ origin @ joint(q)
    and world_geometry = world_link @ offset. Links are declared parent before child (parent -1
    is the base pose). Joints: fixed, revolute (radians about axis), prismatic (metres along
    axis), spherical (wxyz quaternion); the coordinates concatenate in link order, see
    coordinate_slices.
    """

    def __init__(self, parents, joints=None, axes=None, origins=None, part_links=None, offsets=None):
        parents = np.asarray(parents)
        if parents.ndim != 1 or parents.size == 0 or not np.issubdtype(parents.dtype, np.integer):
            raise ValueError("parents must be a nonempty vector of integer link indices (-1 for the base)")
        if np.any(parents < -1) or np.any(parents >= len(parents)):
            raise ValueError("parent link index out of range")
        self.num_links = len(parents)
        joints = ["fixed"] * self.num_links if joints is None else [str(joint) for joint in joints]
        if len(joints) != self.num_links:
            raise ValueError("joints must name one joint type per link")
        for joint in joints:
            if joint not in JOINT_SIZES:
                raise ValueError(f"unknown joint type {joint!r}; expected one of {sorted(JOINT_SIZES)}")
        axes = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (self.num_links, 1)) if axes is None else np.ascontiguousarray(axes, dtype=np.float32)
        if axes.shape != (self.num_links, 3):
            raise ValueError(f"axes must have shape ({self.num_links}, 3)")
        origins = _identities(self.num_links) if origins is None else _rows(origins, self.num_links, "origins")
        part_links = np.arange(self.num_links) if part_links is None else np.asarray(part_links)
        if part_links.ndim != 1 or not np.issubdtype(part_links.dtype, np.integer):
            raise ValueError("part_links must be a vector of integer link indices")
        self.num_parts = len(part_links)
        offsets = _identities(self.num_parts) if offsets is None else _rows(offsets, self.num_parts, "offsets")
        self._model = load_core().RigModel(
            np.ascontiguousarray(parents, dtype=np.int64), joints, axes, origins,
            np.ascontiguousarray(part_links, dtype=np.int64), offsets,
        )
        self.joints = tuple(joints)
        self.num_coordinates = self._model.num_coordinates
        self.coordinate_slices = tuple(
            slice(offset, offset + JOINT_SIZES[joint]) for offset, joint in zip(self._model.coordinate_offsets, joints)
        )
        self._default_coordinates = np.zeros(self.num_coordinates, dtype=np.float32)
        for joint, coordinates in zip(joints, self.coordinate_slices):
            if joint == "spherical":
                self._default_coordinates[coordinates.start] = 1

    @property
    def default_coordinates(self):
        """Neutral joint state: zero scalar joints and identity quaternions."""
        return self._default_coordinates.copy()

    @classmethod
    def from_assembly(cls, assembly, joints=None):
        """One link per part at its rest frame (relative to the enclosing part when the loader
        recorded one); joints maps part names to (type, axis) to articulate them. At the
        default coordinates the rig reproduces what Renderer.set_transform renders."""
        num_parts = assembly.num_parts
        parents = np.array([assembly.part_parent(index) for index in range(num_parts)], dtype=np.int64)
        rests = [_homogeneous(assembly.part(index)[1]) for index in range(num_parts)]
        origins = np.empty((num_parts, 3, 4), dtype=np.float32)
        for index, rest in enumerate(rests):
            origins[index] = (rest if parents[index] < 0 else np.linalg.inv(rests[parents[index]]) @ rest)[:3]
        types = ["fixed"] * num_parts
        axes = np.tile(np.array([0.0, 0.0, 1.0]), (num_parts, 1))
        if joints:
            names = [assembly.object_name(assembly.part(index)[0]) for index in range(num_parts)]
            for name, (joint, axis) in joints.items():
                if name not in names:
                    raise ValueError(f"no part named {name!r}; the parts are {names}")
                types[names.index(name)] = joint
                axes[names.index(name)] = axis
        return cls(parents, types, axes=axes, origins=origins)

    def evaluate(self, base=None, coordinates=None, *, out=None):
        """Geometry-to-world poses (num_parts, 3, 4) for a base pose (3, 4) and joint coordinates
        (num_coordinates,); a (B, 3, 4) base with (B, num_coordinates) coordinates evaluates a
        batch into (B, num_parts, 3, 4). out reuses an output array."""
        base = np.eye(4, dtype=np.float32)[:3] if base is None else np.ascontiguousarray(base, dtype=np.float32)
        batched = base.ndim == 3
        if coordinates is None:
            coordinates = np.broadcast_to(self._default_coordinates, (base.shape[0], self.num_coordinates)) if batched else self._default_coordinates
        coordinates = np.ascontiguousarray(coordinates, dtype=np.float32)
        shape = (base.shape[0], self.num_parts, 3, 4) if batched else (self.num_parts, 3, 4)
        out = _output(out, shape)
        self._model.evaluate(base, coordinates, out)
        return out

    def sample(self, base_open, base_close, coordinates_open, coordinates_close, num_samples, *, out=None, close_out=None):
        """Motion across a shutter: (num_samples, num_parts, 3, 4) poses at the times
        (s + 0.5) / num_samples and the (num_parts, 3, 4) shutter-close poses, ready for
        Renderer.set_motion_transforms. The base follows the constant-twist screw between the
        two poses; the coordinates interpolate before the kinematics, so a revolute joint keeps
        every revolution in between."""
        out = _output(out, (int(num_samples), self.num_parts, 3, 4))
        close_out = _output(close_out, (self.num_parts, 3, 4))
        self._model.sample(
            np.ascontiguousarray(base_open, dtype=np.float32), np.ascontiguousarray(base_close, dtype=np.float32),
            np.ascontiguousarray(coordinates_open, dtype=np.float32), np.ascontiguousarray(coordinates_close, dtype=np.float32),
            int(num_samples), out, close_out,
        )
        return out, close_out

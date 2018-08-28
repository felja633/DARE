import numpy as np
from pathlib import Path

def load_demo_data(path):

    print("load point clouds from ", path)
    coordinates = []
    features = []
    for f in sorted(Path(path).glob("*.txt")):
        v = np.loadtxt(str(f), np.float32)
        coordinates.append(v[:, 0:3].transpose())
        features.append(v[:, 3:].transpose())

    return coordinates, features

def u8_rgb_as_float(rgb):

    rgb = rgb.astype(np.uint32)
    rgb = rgb[0, :] * 65536 + rgb[1, :] * 256 + rgb[2, :]
    rgb.dtype = np.float32
    rgb = np.expand_dims(rgb, axis=0)

    return rgb

def dump_ply(points, colors, path):

    points = np.concatenate(points, axis=1).astype(np.float32)
    colors = np.concatenate(colors, axis=1).astype(np.uint8)
    colors = u8_rgb_as_float(colors)

    N = points.shape[1]

    with open(path, "w") as f:
        print("ply", file=f)
        print("format binary_little_endian 1.0", file=f)
        print("element vertex %d" % N, file=f)
        print("property float x", file=f)
        print("property float y", file=f)
        print("property float z", file=f)
        print("property uchar diffuse_red\n", file=f)
        print("property uchar diffuse_green\n", file=f)
        print("property uchar diffuse_blue\n", file=f)
        print("property uchar unused\n", file=f)
        print("end_header", file=f)

        data = np.concatenate((points, colors), axis=0).transpose()
        data.tofile(f)

def dump_pcd(points, colors, path):

    points = np.concatenate(points, axis=1).astype(np.float32)
    colors = np.concatenate(colors, axis=1).astype(np.uint8)
    N = points.shape[1]

    colors = u8_rgb_as_float(colors)

    with open(path, "w") as f:
        print("VERSION .7", file=f)
        print("FIELDS x y z rgb", file=f)
        print("SIZE 4 4 4 4", file=f)
        print("TYPE F F F F", file=f)
        print("COUNT 1 1 1 1", file=f)
        print("WIDTH %d" % N, file=f)
        print("HEIGHT 1", file=f)
        print("VIEWPOINT 0 0 0 1 0 0 0", file=f)
        print("POINTS %d" % N, file=f)
        print("DATA binary", file=f)

        data = np.concatenate((points, colors), axis=0).transpose()
        data.tofile(f)


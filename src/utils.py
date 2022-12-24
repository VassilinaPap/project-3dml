import numpy as np
import open3d as o3d

# TUM Prof. Dai and TAs code #
def read_as_3d_array(fp, fix_coords=True):
    """ Read binary binvox format as array.
    Returns the model with accompanying metadata.
    Voxels are stored in a three-dimensional numpy array, which is simple and
    direct, but may use a lot of memory for large models. (Storage requirements
    are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
    boolean arrays use a byte per element).
    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    # return Voxels(data, dims, translate, scale, axis_order)
    return data

def save_voxel_grid(path, voxel_grid):

    voxels = []

    # TODO: Likely a more efficient way to do this using NumPy API
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[0]):
                if voxel_grid[i,j,k] >= 0.5:
                    voxel = [i, j, k]
                    voxels.append(voxel)


    pcd = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(np.array(voxels, dtype=int))
    pcd.points = points
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(len(points), 3)))
    o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 1)
    o3d.io.write_voxel_grid(path, o3d_voxel_grid)
    
def read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = [int(i) for i in fp.readline().strip().split(b' ')[1:]]
    translate = [float(i) for i in fp.readline().strip().split(b' ')[1:]]
    scale = [float(i) for i in fp.readline().strip().split(b' ')[1:]][0]
    line = fp.readline()
    return dims, translate, scale

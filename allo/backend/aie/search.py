from itertools import product
import numpy as np
import copy


def build_empty_grid(shape):
    if not shape:
        return None
    if len(shape) == 1:
        return [None] * shape[0]
    return [build_empty_grid(shape[1:]) for _ in range(shape[0])]


def count_base_elements(obj):
    if not isinstance(obj, list):
        return 1
    return sum(count_base_elements(x) for x in obj)


def slice_along_axis(grid, axis, fixed_value):
    result = []

    def helper(subgrid, dim):
        if not isinstance(subgrid, list):
            result.append(subgrid)
            return
        if dim == axis:
            helper(subgrid[fixed_value], dim + 1)
        else:
            for child in subgrid:
                helper(child, dim + 1)

    helper(grid, 0)
    return result


def chain_primitive(node_name_lists: list[str]):
    mapping_primitives = []
    base = node_name_lists[0]
    for k in range(1, len(node_name_lists)):
        mapping_primitives.append(("chain", [base, node_name_lists[k]]))
        base += f"-{node_name_lists[k]}"
    return mapping_primitives, base


def chain_on_axes(axes, c_tiles, grid, grid_size):
    mapping_primitives = []
    sizes = [(s if i != axes else s // c_tiles) for i, s in enumerate(grid_size)]
    new_grid = build_empty_grid(sizes)
    for outer_idx in product(*(range(s) for s in sizes)):
        nodes = []
        for offset in range(c_tiles):
            idx = list(outer_idx)
            idx[axes] = idx[axes] * c_tiles + offset
            elem = grid
            for idx_ in idx:
                elem = elem[idx_]
            nodes.append(elem)
        chains, name_ = chain_primitive(nodes)
        mapping_primitives.extend(chains)
        target = new_grid
        for i in outer_idx[:-1]:
            target = target[i]
        target[outer_idx[-1]] = name_
    return new_grid, mapping_primitives, sizes


def bundle_on_axes(axes, b_tiles, grid, grid_size):
    mapping_primitives = []
    sizes = [(s if i != axes else s // b_tiles) for i, s in enumerate(grid_size)]
    if 0 in sizes:
        return None, None, sizes
    new_grid = build_empty_grid(sizes)
    size_ = grid_size[axes] // b_tiles
    for i in range(size_):
        nodes = []
        for offset in range(b_tiles):
            nodes.append(slice_along_axis(grid, axes, i + offset * size_))
        mapping_primitives.append(("bundle", nodes))
        for outer_idx in product(*(range(s) for s in sizes)):
            elem = grid
            for idx_ in outer_idx:
                elem = elem[idx_]
            target = new_grid
            for i in outer_idx[:-1]:
                target = target[i]
            target[outer_idx[-1]] = f"{elem}x{size_}"
    return new_grid, mapping_primitives, sizes


def search(reduce_axes, parallel_axes, grid, grid_size):
    mappings = []
    factors = [1, 2, 4, 8, 16]
    axes_list = reduce_axes + parallel_axes

    def reduce_graph(grid_, grid_size_, mapping_primitives, axes_id):
        if np.prod(grid_size_) <= 16:
            print(np.prod(grid_size_))
            mappings.append(mapping_primitives)
            return
        if axes_id == len(axes_list):
            return
        axes = axes_list[axes_id]
        for t_size in factors:
            if grid_size_[axes] // t_size < 2:
                reduce_graph(grid_, grid_size_, mapping_primitives, axes_id + 1)
                continue
            if axes in reduce_axes:
                grid__, primitives, sizes = chain_on_axes(
                    axes,
                    grid_size_[axes] // t_size,
                    grid_,
                    grid_size_,
                )
            else:
                grid__, primitives, sizes = bundle_on_axes(
                    axes,
                    grid_size_[axes] // t_size,
                    grid_,
                    grid_size_,
                )
            mapping_primitives_: list = copy.deepcopy(mapping_primitives)
            mapping_primitives_.extend(primitives)
            reduce_graph(grid__, sizes, mapping_primitives_, axes_id + 1)

    reduce_graph(grid, grid_size, [], 0)

    return mappings

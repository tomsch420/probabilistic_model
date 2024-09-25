from jax.experimental.sparse import BCOO

def copy_bcoo(x: BCOO) -> BCOO:
    return BCOO((x.data, x.indices), shape=x.shape, indices_sorted=x.indices_sorted,
                unique_indices=x.unique_indices)
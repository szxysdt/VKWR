def import_all_ops():
    from vkwr._ops import rwkv7_ops, sampling_ops
    from vkwr._ops.v1 import import_all_v1_ops

    import_all_v1_ops()

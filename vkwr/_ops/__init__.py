def import_all_ops():
    from vkwr._ops import common_ops, sampling_ops  # noqa: F401
    from vkwr._ops.v1 import import_all_v1_ops
    from vkwr._ops.v2 import import_all_v2_ops
    from vkwr._ops.v2_5 import import_all_v2_5_ops

    import_all_v1_ops()
    import_all_v2_ops()
    import_all_v2_5_ops()

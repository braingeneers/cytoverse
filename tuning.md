╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name status pq_m pq_k num_partitions n_probe iter total time (s) accuracy total_wire_mb unique_partitions combined_score │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ IVFPQTrainable_afdc7_00000 TERMINATED 16 512 128 1 1 4133.73 0.8935 435.263 127 0.458237 │
│ IVFPQTrainable_afdc7_00001 TERMINATED 32 512 512 1 1 4154.59 0.88 697.106 452 0.182894 │
│ IVFPQTrainable_afdc7_00002 TERMINATED 32 256 256 2 1 2208.65 0.901 786.558 255 0.114442 │
│ IVFPQTrainable_afdc7_00003 TERMINATED 16 128 512 1 1 315.791 0.8735 395.849 462 0.477651 │
│ IVFPQTrainable_afdc7_00004 TERMINATED 8 256 128 1 1 2575.28 0.8885 263.214 128 0.625286 │
│ IVFPQTrainable_afdc7_00005 TERMINATED 16 256 512 1 1 748.049 0.882 394.136 460 0.487864 │
│ IVFPQTrainable_afdc7_00006 TERMINATED 16 128 128 2 1 237.487 0.889 435.263 127 0.453737 │
│ IVFPQTrainable_afdc7_00007 TERMINATED 16 256 512 2 1 739.813 0.896 425.838 497 0.470162 │
│ IVFPQTrainable_afdc7_00008 TERMINATED 8 256 256 2 1 3174.88 0.8925 260.13 253 0.63237 │
│ IVFPQTrainable_afdc7_00009 TERMINATED 8 256 512 3 1 3420.66 0.8935 259.615 505 0.633885 │
│ IVFPQTrainable_afdc7_00010 TERMINATED 8 128 512 2 1 1601.08 0.882 255.503 497 0.626497 │
│ IVFPQTrainable_afdc7_00011 TERMINATED 16 128 512 3 1 1631.86 0.891 434.406 507 0.456594 │
│ IVFPQTrainable_afdc7_00012 TERMINATED 32 512 256 2 1 3030.9 0.8995 786.558 255 0.112942 │
│ IVFPQTrainable_afdc7_00013 TERMINATED 16 128 512 3 1 1064.33 0.892 427.552 499 0.464448 │
│ IVFPQTrainable_afdc7_00014 TERMINATED 16 256 512 1 1 1365.47 0.8745 394.136 460 0.480364 │
│ IVFPQTrainable_afdc7_00015 TERMINATED 8 512 128 1 1 1664.92 0.88 259.101 126 0.620899 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

INFO:**main**:Best configuration found:
INFO:**main**: Config: {'pq_m': 8, 'pq_k': 256, 'num_partitions': 512, 'n_probe': 3, 'total_num_embeddings': 23000000, 'train_embeddings_ref': ObjectRef(00ffffffffffffffffffffffffffffffffffffff0100000001e1f505), 'test_embeddings_ref': ObjectRef(00ffffffffffffffffffffffffffffffffffffff0100000002e1f505), 'y_train_ref': ObjectRef(00ffffffffffffffffffffffffffffffffffffff0100000003e1f505), 'y_test_ref': ObjectRef(00ffffffffffffffffffffffffffffffffffffff0100000004e1f505)}
INFO:**main**: Metrics: accuracy=0.893, total_wire_mb=259.6MB, unique_partitions=505

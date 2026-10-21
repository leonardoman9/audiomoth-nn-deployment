# Model package

Everything the firmware needs from one trained model. The architecture is fixed (WrenNet: 32 kHz, 3 s windows, 64
learned filters, int8 CNN backbone, GRU 64 + attention); a package can have any number of classes (2–255).

```
<package>/
  *.tflite                     int8 backbone (input [1,1,301,64], output [1,1,301,32]); exactly one file
  model_config.json            classes, names, negative class, daily window, hop, threshold (read by the tests and tools)
  generated/
    model_config.h             the same, for the firmware   (firmware/bird/gen_model_config.py)
    bird_frontend_tables.h     learned filterbank tables    (tools/gen_frontend_tables.c)
    bird_tail_weights_i8.h     GRU, projection, attention, classifier weights   (firmware/bird/gen_bird_tail.py)
    tail_dims.h                frames per window            (firmware/bird/gen_bird_tail.py)
  test/                        optional: one known window for the QEMU test (firmware/bird/gen_pipeline_vec.py)
    pipe_vec.cc  specs_one.npz
```

The build scripts take a package folder, or a `.tflite` inside one:
`firmware/app/build_app.sh TFLM_ROOT path/to/package`, `tests/sim/build_sim.sh TFLM_ROOT path/to/package`,
`tests/qemu/run_qemu_pipeline.sh TFLM_ROOT python3 path/to/package`. The Python tests read `MODEL_PKG`
(default: this folder).

## This package: WrenNet birds, 8 species + non-bird

Checkpoint `best_distillation_model.pt` (WrenNet repository), `bird_backbone_int8.tflite` sha256 `6f77fa86…`.

Class order = the model's output indices, which follow `allowed_bird_classes` in the training config: 0 Bubo bubo,
1 Certhia familiaris, 2 Apus apus, 3 Certhia brachydactyla, 4 Emberiza cia, 5 Lophophanes cristatus, 6 Periparus ater,
7 Poecile montanus, 8 non-bird. It is **not** the order of the "Class names" line of the training log, which comes from
the soft-label metadata and only labels the report. Evidence:
1. the per-class test supports of the training report match 15 % of the file counts of the dataset folders only in
   this order;
2. the checkpoint run on real recordings of all 8 species predicts exactly these indices (docs/VERIFICATION.md,
   section 8).

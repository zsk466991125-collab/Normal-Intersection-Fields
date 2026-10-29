# Release verification

These checks validate packaging and execution. The accuracy tables in the
README are the paper's reported results; full benchmarks were not rerun.

- All 36 legacy configurations parsed and built successfully: 32 paired
  baseline/NIF configurations and four frozen-carrier configurations.
- Every legacy configuration uses two images/GPU and lr 0.0001.
- The four two-stage carrier families loaded their existing NIF checkpoints
  strictly, ran real-image inference, and produced finite training losses
  and nonzero NIF gradients.
- LSKNet-S + FAA Head and PKINet-v2-S frozen checks confirmed no carrier
  gradients or normalization-buffer changes and a successful NIF update.
- Numerical comparison with the experiment implementation gave maximum
  absolute error 0 for field predictions, reliability, refined boxes,
  geometric losses, and final detections on the checked example.
- A four-GPU legacy training run completed one iteration with two images
  per GPU. All 411 loaded LSKNet-S + FAA Head carrier tensors remained
  bitwise identical after the optimizer step.
- Ultralytics loaded a converted NIF checkpoint, completed half-precision
  inference, and backpropagated only through NIF. A native four-GPU run
  completed one epoch/iteration, validation, checkpoint save, and reload.
- O2-RTDETR loaded all 791 carrier entries, initialized only the 12 new NIF
  state entries, and passed inference/backward checks. Native four-GPU
  training completed one epoch/iteration with two images per GPU.
- DOTA, FAIR XML, YOLO DOTA, and native O2 DOTA submission writers produced
  valid archive layouts on one-image checks. The FAIR raw ZIP preparation
  helper also completed native multiscale splitting on a small input.

The distributed checks use a tiny dataset view outside this repository.
They establish that the launchers and gradients work; they measure no
benchmark accuracy. No test datasets or generated checkpoints are included.

The legacy checks used Python 3.9, PyTorch 1.13.1/CUDA 11.7, MMCV 1.7.0,
MMDetection 2.28.2, and timm 1.0.29. O2 used a separate dependency path with
MMCV 2.1.0, MMDetection 3.3.0, and MMEngine 0.10.7.

[Machine-readable details](verification.json)

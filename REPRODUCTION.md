# Reproduction

## Rebuild figures or reanalyse recorded measurements

For Figures 2, 3 and 5, place the accompanying paper's source_data directory at the repository root and install NumPy and Matplotlib. From that root, run the following command. The PDFs are written beside source_data.

    python figure_sources/plot_figures.py

To recompute tables from the completed board campaign, unpack the verified board experiment records under reports/scientific_reports_validation_1/board_results/reports. Place the frozen joint240_rows.jsonl and contract_v1.json in reports/scientific_reports_validation_1. With NumPy and OpenCV installed, run:

    python -X utf8 -B scripts/scientific_reports_validation_1/analyze_joint_reproduce.py --output-subdir recomputed_run

Use a fresh output name for each run. The analysis reads existing records and performs no board I/O. The separate OSDaR23 audit also needs the external cohort manifest, stored PC predictions and original OpenLABEL annotations, with paths specified in audit_external.py.

## Rebuild the runtime and run the reference

The model compiler and PC reference use the versions recorded in yolo26_optimization_8/requirements_pc.txt. Obtain the official YOLO26n checkpoint separately and place it at yolo26_optimization_8/reproduction_inputs/yolo26n.pt. The frozen checkpoint SHA-256 is 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef.

The rectangular preparation and compilation entry points are scripts/railway_person_optimization/prepare_railptq.py and recompile_models.py. They use the frozen split and calibration inputs from the railway experiment package. The prepared candidates are expected under output/railway_person_optimization, with separate directories for the 384 × 640 global graph and 256 × 1280 refinement graph. The 736 × 1280 control belongs to the NCNN experiment.

The native candidate-v3 files and their source hashes are in railway_system_optimization_1/native_sources/candidate_v3. The ARM build script uses the board's GCC toolchain, Cortex-A9 NEON and OpenMP, with floating-point contraction disabled. For the Windows PC build, set RAILWAY_CXX to the matching MinGW g++ executable and run scripts/railway_system_optimization_1/build_pc_v3.py.

The independently reproduced NCNN FP32 global control uses the packaged code in reproduction/ncnn_control, its locked requirements, the separately retained model and reference payload, and the official COCO val2017 data:

    python reproduction/ncnn_control/reproduce_control.py --payload PATH_TO_REFERENCE_PAYLOAD --coco-root PATH_TO_COCO --output PATH_TO_NEW_OUTPUT

## Board and external data

Vivado 2026.1, the TUL PYNQ-Z2 board definition and AMD IP are external dependencies. The accepted design uses the XC7Z020 and a 100 MHz PL clock. Use yolo26_optimization_8/hardware/rtl with scripts/optimization_8/build_optimization_8_resident.tcl. The Tcl entry expects OPTIMIZATION_8_PROJECT_ROOT, OPTIMIZATION_8_SOURCE_SNAPSHOT and OPTIMIZATION_8_RUN_ID. The snapshot directory must contain rtl and the vendor board definition under board_repo. The accepted firmware identity is recorded in the firmware manifest.

The original board scripts expect the frozen artifacts under /home/xilinx/jupyter_notebooks in the layouts declared in their source. They attach to the accepted loaded hardware and validate its identity. The repository by itself is insufficient to replay the original board measurements. A replay also needs the accepted firmware, compiled model assets, matching native library, and the frozen image and PC-reference bundles.

Get the image datasets from their original sources under their respective terms:

- [RailGoerl24](https://data.fid-move.de/dataset/railgoerl24)
- [OSDaR23](https://doi.org/10.57806/9mv146r0)
- [COCO](https://cocodataset.org/)

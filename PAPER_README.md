# YOLO26n ARM–FPGA railway person detection

This repository contains the research software for the PYNQ-Z2 experiments in *Numerically consistent ARM–FPGA execution of a quantized detector reveals scale-dependent acceleration and railway detection limits*. The implementation evaluates the frozen Ultralytics YOLO26n on decoded railway images using a sequential INT8 convolution engine and ARM host processing.

| Directory | Contents |
| --- | --- |
| yolo26_optimization_8 | Convolution RTL, activation cache, AXI traffic counter, graph compiler and native execution interface |
| railway_system_optimization_1 | Frozen candidate-v3 native C++ source used for the rectangular experiments |
| railway_person_optimization and railway_budget_research | Rectangular inference, strip selection, region-score policy and person-box fusion |
| railway_sensors_validation_1 | NCNN comparator and slicing wrappers |
| scripts/scientific_reports_validation_1 | Board workloads, PC reference, offline analysis and external failure audit |
| scripts/railway_distance_1 | Exploratory OSDaR23 range and monocular-depth analysis |
| figure_sources | Plots for the manuscript's retention, application tradeoff and external failure figures |

See [Reproduction](REPRODUCTION.md) for dependencies, artifact placement and commands. Input images, model binaries, routed FPGA images and experiment records are distributed separately under their applicable access terms. The railway experiments report detection against supplied boxes in public datasets.

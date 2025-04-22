# GABMIL

This repository contains code and models for a spatially-aware Multiple Instance Learning model.

## Table of Contents

- [Overview](#overview)
- [Framework](#framework)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [Reference](#reference)

## Overview

Global ABMIL (GABMIL) enhances Attention-Based Deep MIL (ABMIL) for whole slide image classification by modeling patch interactions without adding significant computational cost. Using a lightweight Spatial Information Mixing Module (SIMM), GABMIL improves performance by up to 7 percentage point in AUPRC and 5 percentage point in Kappa score over ABMIL, while staying much more efficient than Transformer-based methods like TransMIL.

## Framework

<p align="left">
  <img src="framework.png" alt="Framework">
  <br>
  <em>Figure 1: An overview of our GABMIL method. We first divide the input WSI into patches and extract their corresponding features using a pretrained model. The Spatial Information Mixing Module (SIMM) then integrates spatial information into the feature representations. Finally, the ABMIL model predicts the slide-level label.</em>
</p>

<p align="left">
  <img src="simm.png" alt="SIMM">
  <br>
  <em>Figure 2: (a) Illustration of the SIMM (BOTH configuration). Patch features are repositioned according to their original spatial arrangement. The BLOCK and GRID attention modules are then applied sequentially to integrate spatial information into the feature representations. (b) The BLOCK attention module captures spatial information within partitioned windows using a MLP layer. (c) The GRID attention module models spatial information within each partitioned grid using a MLP layer.</em>
</p>

## Results



## Acknowledgements


## Reference

Please consider citing the following paper if you find our work useful for your project.


```
@InProceedings{,
  title = {},
  author = {},
  booktitle = {},
  pages = {},
  year = {2025},
  volume = {},
  series = {},
  month = {},
  publisher = {},
}
```

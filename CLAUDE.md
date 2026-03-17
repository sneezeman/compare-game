# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository stores visual comparison data for 3D reconstruction results. It contains GIF animations showing reconstruction outputs across training epochs from multiple viewpoints.

## Data Layout

Each subdirectory represents a comparison between a source and reference reconstruction, named as:
`{source}_from_{reference}_{epoch}`

Each directory contains three GIF files (`all_epochs_view0.gif`, `all_epochs_view1.gif`, `all_epochs_view2.gif`) showing the reconstruction progress from different camera angles.

## Naming Convention

Directory names encode experiment metadata:
- Sample ID (e.g., `NM0029_HT_100nm`)
- Tomogram index (e.g., `T001`, `T014`)
- Scan number (e.g., `0001`)
- Suffix `rec` indicates reconstructed data
- Trailing number is the epoch identifier

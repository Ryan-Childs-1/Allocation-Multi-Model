# Allocation Split Expert Model Report

## What the model does

This Streamlit app uses a two-stage NumPy inference system for allocation rows. Allocate and Review rows are handled by separate model pairs. Each row is first classified into an allocation group, then a neural regressor predicts the number of FLMs to allocate. Review rows are additionally ranked by need and allocated until the modeled DC availability pool is exhausted. Below-FLM allocation is allowed only when the remaining DC quantity is less than one FLM.

## Approved model inputs

Class Name, Line Name, Site, MIL, FLM, Cost, L30, D30, D60, LW, TTM, Supply, Dc Avail, Rank, Proj. Demand, Alloc. Rec., Flag

## Architecture

Classifier hidden layers: [512, 256, 128]


Regressor hidden layers: [768, 384, 192, 96]


Allocate threshold: 0.4


Review threshold: 0.62

## Top overall feature groups

- Line Name category signal: 33.727%

- Site category signal: 25.386%

- Class Name category signal: 24.972%

- Dc Avail Bucket category signal: 2.109%

- Rank category signal: 2.099%

- Flag category signal: 2.080%

- dc_flms: 0.274%

- supply_to_proj: 0.274%

- d60_gap: 0.273%

- dc_flms_capped: 0.271%

- zero_supply: 0.268%

- FLM: 0.268%

- supply_to_d60: 0.267%

- supply_to_d30: 0.266%

- MIL: 0.264%

- rec_flms: 0.264%

- Supply: 0.263%

- D30: 0.261%

- rank_score: 0.261%

- l30_to_d30: 0.260%

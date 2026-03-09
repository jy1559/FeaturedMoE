# FeaturedMoE_ProtoX Quick Guide

## Core Idea
- Prototype-first routing: session prototype mixture `pi` is estimated first.
- `pi` conditions both stage allocation and stage expert routing.
- Stability knobs: `stage_weight_floor`, `stage_delta_scale`, and prototype temperature annealing.

## Key Configs
- Prototype: `proto_num`, `proto_dim`, `proto_top_k`, `proto_temperature_start/end`.
- Prototype regularization: `proto_usage_lambda`, `proto_entropy_lambda`.
- Conditioning: `proto_router_use_hidden`, `proto_router_use_feature`, `proto_pooling`.
- Stage merge: `protox_stage_merge_mode` (`serial_weighted` / `parallel_weighted`).

## Logs
- Runtime logs include prototype usage KL, prototype entropy KL, and stage allocation statistics.

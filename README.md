# HHB-FL
Implementation of HHB-FL: a blockchain-based federated learning framework with hierarchical homomorphic encryption and differential privacy.

AIS Data Set  https://www.fisheries.noaa.gov/inport/item/55360
HHB-FL is a research-oriented framework for privacy-preserving federated learning (FL), designed for IoMT and other distributed environments.
It integrates hierarchical homomorphic encryption (Paillier + CKKS), differential privacy, and a blockchain-based smart-contract aggregator for on-chain verification and auditability.
The repository provides both single-node simulation scripts and Docker-based distributed deployment to reproduce experiments and evaluate performance or security under various attack models.

Initialization & Configuration — automatic setup of keys, CKKS parameter sets, and DP constants.

Data Cleaning & Preprocessing — scripts for dataset preparation, normalization, and non-IID partitioning.

Local Training — PyTorch-based SGD training with L₁ clipping and Laplace noise injection for differential privacy.

Federated Simulation — Python scripts for multi-client FL with quantization, encryption, and aggregation rounds.

Smart Contract for On-chain Aggregation & Verification — verifies signatures, prevents replay, checks parameter consistency, and performs Paillier homomorphic aggregation plus CKKS bias logging.

CKKS Parameter Set — predefined configurations (e.g., add_only_8k, balanced_16k, high_prec_32k) included inside the code, supporting encrypted floating-point operations and packed bias aggregation.

Encryption & Decryption Utilities — Paillier integer-domain and CKKS floating-point encryption functions.

Docker-based Distributed FL — a ready-to-run Compose environment (1 contract + 10 clients).

Blockchain Upload — optional scripts for writing transaction or model hashes to a blockchain ledger.

Attack Modules — examples of replay, poisoning, and inference attacks for robustness testing.



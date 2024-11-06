# Package Aware DRL Scheduler

## Overview
Serverless edge computing, especially in Function-as-a-Service (FaaS) models, has grown in popularity as it enables developers to run functions without managing infrastructure. However, efficient task scheduling to optimize resources and reduce delays remains a challenge. This simulator enables testing package-aware and dependency-aware scheduling approaches using DRL techniques.
This repository builds upon the original [MFS repository by hanaforoosh](https://github.com/hanaforoosh/MFS).

## Key Features
- **Package-Aware Scheduling**: Reduces cold start delays by prioritizing containers with pre-installed packages.
- **Dependency-Aware Scheduling**: Minimizes delays by co-locating dependent functions.
- **Reward Function for DRL**: Incorporates a custom reward function that promotes lower execution times, efficient package installation, and optimal dependency management.
- **DRL Algorithms**: Adapts scheduling decisions dynamically using Deep SARSA and DQN.
- **Extensive Metrics**: Provides detailed metrics for analyzing scheduling performance.
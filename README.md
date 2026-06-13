# final-project-submission-mmende29
final-project-submission-mmende29 created by GitHub Classroom

# 5G-NR Signal SNR Classification Project

## Overview
This project implements a deep learning solution for classifying 5G-NR signals based on their SNR levels. It processes raw RF signals and uses a CNN to determine signal quality.


## Signal Specifications
- Signal Type: 5G-NR (New Radio)
- Center Frequency: 628 MHz
- Sampling Rate: 20 MHz
- Bit Depth: 16-bit
- Input Impedance: 50 ohm

## Dataset Details
Source: IEEE Dataport 5G-NR signal collection
File Format: Binary I/Q samples (interleaved 16-bit integers)
Sample Length: 2000 samples per segment
Total Samples: 10,000 segments
Data also found here: https://www.kaggle.com/datasets/siddss/real-world-wireless-communication-dataset?resource=download&select=reading_signaldata.py

## SNR Classification Levels
The model classifies signals into three SNR categories:
- High SNR: 20 dB
- Medium SNR: 10 dB
- Low SNR: 0 dB

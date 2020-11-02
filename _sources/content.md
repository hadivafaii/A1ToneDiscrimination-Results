Content
=======================

This update includes 3 sections:
1. Data processing
2. Classification Analysis
3. Linear Discriminant Analysis
4. Clf analysis (with limited time range) + supervised PCA
5. Neural Net Analysis

## What's new:

Limited best timepoints

    I choose best time points in different intervals and compare performance
    drops.  Overall, limiting the time period leads to a drop in
    performance; however it still remains well above chance level.

Supervised PCA
    
    Unfortunately, the algorithm doesn't work here for two reasons.  First, we
    are limited with number of samples which can be as low as 3 in some cases.
    In this example, the lowesr number of PC dimensions we can have is 3, which
    prevents us from applying this systematically to all the data that comes with
    varying number of sampels for different tasks/experiments.  Another reason is
    in the reservoir paper, this algorithm was applied to a nonlinear, high dim
    expansion of the raw signal.  Here we only have the raw signal and applying
    PCA to that to achieve dimensionality reduction hurts performance a lot.
    
    For these reasons, the supervised PCA algorithm does not seem appropriate for
    studying the dimensioanlity of the Kanold data.
    
Neural Net analysis

    To have something about dimensionalities, I performed multi-layer perceptrion
    analysis.  Please see the results in the last MLP section.
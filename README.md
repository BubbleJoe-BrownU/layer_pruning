# layer_pruning

This project aims to reproduce the result presented in the paper, *The Unreasonable Ineffectiveness of The Deeper Layers*. After replicating the result, next steps include exploring how we can improve on top of it.

## Method
The authors of the paper present a simple algorithm for layer pruning a Transformer-based model:
0. Pick a number of layers to prune n.
1. Compute the angular distance (defined in `utils.py`) of $d(x^{(l)}, x^{(l+n)})$ between the **input** to layer $l$ and the input to layer $l + n$ on a neutral pretraining dataset or on a dataset representative of a downstream task of interest.
2. Find the layer $l^{*}$ that minimizes that distance:
$$
l^{*}(n) \equiv \text{arg} \text{min}_t d(x^{(l)}, d^{(l+n)})
$$
3. Drop layers $l^{*}$ to $l^{*}+n-1$; connect the old input to layer $l^{*}$ to the old $(l^{*}+n)$th layer block.
4. (Optionally) heal the mismatch at layer $(l^{*}+n)$ with a small amount of fine tuning on a neutral pretraining dataset or particular dataset of interest.

## Timeline (estimated)
1. First Three Weeks: Establish the code that's able to detect the optimal layer, starting from which the model will be pruned $n$ layers, prune $n$ layers from the model, and perform PEFT finetuning for the result model.
2. Next Two Weeks: Run experiments with popular LLMs, including Llama-2-7b, Mistral-7b, Gemma-2b, Gemma-7b, to see the effect of this layer pruning algorithm on up-to-date models.
3. Next Eight Weeks: Explore and design improvements upon the existing layer pruning algorithm and run experiments.
4. Next Two Weeks: Wrap up the entire project and write a manuscript on it.

Progress Bar:
```
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
```
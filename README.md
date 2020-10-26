# UNet.jl

This pacakge provides a generic UNet implemented in Julia using Flux. Originally based on https://github.com/DhairyaLGandhi/UNet.jl but heavily modified. 

## Further Reading
The package is an implementation of the [paper](https://arxiv.org/pdf/1505.04597.pdf), and all credits of the model itself go to the respective authors.

## Usage

See runtests.jl to see how to overfit a single image, also train.jl for a generic training script.

* Input: 
    * ![GitHub Logo](/test/testdata/input.png)
* Target: 
    * ![GitHub Logo](/test/testdata/target.png)
* Prediction: 
    * ![GitHub Logo](/test/testdata/prediction.png)
* Training: 
    * ![GitHub Logo](/test/testdata/training.gif)
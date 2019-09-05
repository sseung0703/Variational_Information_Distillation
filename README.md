# Variational_Information_DIstillation

## Abstract
- Pros
- Cons

## Requirements
* python==3.x
* tensorflow>=1.13.0
* Scipy
## How to run

## Experiment results
My experimental results are higher than the paper. I found that It is tough to make such a low performance like paper.

For this, I removed gamma and regularization for batch normalization, and modify hyper-parameter to make training unstable.

   Methods  | Last Accuracy | Paper Accuracy | Last Accuracy |  Paper Accuracy 
:----------:| :-----------: | :-----------:  | :------------:|  :-------------: 
Student     |     91.22     |     90.72      | - | - 
Teacher     |     94.98     |     94.26      | - | - 
Soft-logits | - | 91.27| - | - 
FitNet      | - | 90.64| - | - 
AT          | - | 91.60| - | - 
VID         | - | 91.85| - | - 
<p align="center">
  <img src="plots.png" width="600"><br>
  <b>Experimental results of full dataset</b>  
</p>

## TO DO
- Train student network using comparative methods.
- edit README

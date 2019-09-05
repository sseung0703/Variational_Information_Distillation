# Variational_Information_DIstillation
Project of Reproducing "VID" involved in https://github.com/rp12-study/rp12-hub


## Abstract
- Pros
- Cons

## Requirements
* python==3.x
* tensorflow>=1.13.0
* Scipy
## How to run

## Note that
- I found the author's code at https://github.com/ssahn0215/variational-information-distillation. However I'll not refer it, cause I want to check reproducibility of the paper.
- My experimental results are higher than the paper. I found that It is tough to make such a low performance like paper. For this, I removed gamma and regularization of batch normalization, and modify hyper-parameters to make training unstable.
- The authors said "We choose four pairs of intermediate layers similarly to [31], each of which is located at the end of a group of residual blocks." but there are only three groups of residual blocks in WResNet. So I sense one feature map after the first convolutional layer.
- I'll not follow the author's configuration for comparative methods. Because their modification look somewhat awkward, unfair and not coinside with the proposed ways. Also, I think that for fair comparison should not modify the original author configutation whether good or not. It means that I'll only reprocude the author's method, VID.

## Experiment results
<table>
  <tr>
    <th></th><th colspan="2">Full Dataset</th><th colspan="2">20% Dataset</th><th colspan="2">10% Dataset</th><th colspan="2">2% Dataset</th>
  </tr>
  <tr>
    <td>Methods</td><td>Last Accuracy</td><td>Paper Accuracy</td><td>Last Accuracy</td><td>Paper Accuracy</td><td>Last Accuracy</td><td>Paper Accuracy</td><td>Last Accuracy</td><td>Paper Accuracy</td>
  </tr>
  <tr>
    <td>Student</td>  <td>91.22</td><td>90.72</td><td></td><td>84.67</td><td></td><td>79.63</td><td></td><td>58.84</td>
  </tr>
  <tr>
    <td>Teacher</td>  <td>94.98</td><td>94.26</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td>
  </tr>
  <tr>
    <td>KD(Soft-logits)</td>  <td></td><td>91.27</td><td></td><td>86.11</td><td></td><td>82.23</td><td></td><td>64.24</td>
  </tr>
  <tr>
    <td>FitNet</td>  <td></td><td>90.64</td><td></td><td>84.78</td><td></td><td>80.73</td><td></td><td>68.90</td>
  </tr>
  <tr>
    <td>AT</td>  <td>91.85</td><td>91.60</td><td></td><td>87.26</td><td></td><td>84.94</td><td></td><td>73.40</td>
  </tr>
  <tr>
    <td>VID</td>  <td></td><td>91.85</td><td></td><td>89.73</td><td></td><td>88.09</td><td></td><td>81.59</td>
  </tr>
</table>
<p align="center">
  <img src="plots.png" width="600"><br>
  <b>Experimental results of full dataset</b>  
</p>

## TO DO
- Train student network using comparative methods.
- edit README

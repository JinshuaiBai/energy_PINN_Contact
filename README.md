# Energy-based physics-informed neural network for frictionless contact problems under large deformation

This repository provides Python codes for the paper **Energy-based physics-informed neural network for frictionless contact problems under large deformation**. 

<p align="justify">
The energy-based PINN (aka deep energy method) offers a robust and effective way for modelling solid mechanics problems. By incorporating contact potentials, the energy-based PINN framework can easily simulate contact problems under large deformation. We believe that the power of deep learning techniques should pay more attention to those challenging nonlinear problems. To help those researchers who are interested in this field, this repository provides the open-source energy-based PINN codes for 2D frictionless contact problems under large deformation, including:
</p>

  - Rubber ring contact instability (neo-Hookean under large deformation) (**Section 4.2** in the manuscript)
   ![Image](https://github.com/user-attachments/assets/2588af0d-1282-41bd-b398-f7a2394f0bac)  
    <strong>Fig. 1.</strong> Results for the rubber ring contact instability.

  - Compression of two rubber rings (neo-Hookean under large deformation) (**Section 4.3** in the manuscript)
   ![Image](https://github.com/user-attachments/assets/4d800165-fbdd-4253-a2e5-edab0a183786)  
    <strong>Fig. 2.</strong> Results for the compression of two rubber rings.

This paper has been accepted by **Computer Methods in Applied Mechanics and Engineering**. For more details in terms of implementations and more interesting numerical examples, please refer to our paper.

# Run code
<p align="justify">
Please run the file "main.py" in each case. All the input data is prepared in "Coord.mat". The output (results) are also saved in ".m" files.
</p>

# Paper link
https://doi.org/10.1016/j.cma.2025.117787  
https://arxiv.org/abs/2411.03671

# Enviornmental settings
 - Python 3.10
 - TensorFlow 2.10.0
 - Numpy 1.25.1
 - Scipy 1.11.1

# Cite as
<p align="justify">
[1] J. Bai, Z.Lin, Y. Wang, J. Wen, Y. Liu, T. Rabczuk, Y.T. Gu, X.-Q. Feng, Energy-based physics-informed neural network for frictionless contact problems under large deformation, Computer Methods in Applied Mechanics and Engineering 437 (2025) 117787. 
</p>

# Contact us
For questions regarding the code, please contact:

Dr. Jinshuai Bai: jinshuaibai@gmail.com or bjs@mail.tsinghua.edu.cn  
Prof. YuanTong Gu: yuantong.gu@qut.edu.au  
Prof. Xi-Qiao Feng: xqfeng@tsinghua.edu.cn

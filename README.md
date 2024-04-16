# Learned Regularization for Inverse Problems: Insights from a Spectral Model
In this respository, we document the numerical examples from our paper [Learned Regularization for Inverse Problems: Insights from a Spectral Model](https://arxiv.org/abs/2312.09845).

> [!NOTE]
> If you spot a mistake, encounter any problems, or have any further comments or questions, feel free to send an email to samira.kabri[at]desy.de.

### Content
> #### [Data](#4)
> #### [Requirements](#5)
> #### [Experiments](#6)
> #### [References](#7)
> #### [Citation](#8)


### <a id="3"> Quick Intro </a>


### <a id="4"> Data </a>
In order to run the code smoothly and without to many package dependencies, some pre-saved data is required. The respective folder **spectralData** can be downloaded here:\
[https://syncandshare.desy.de/index.php/s/rSysTmdGAtYAf7r](https://syncandshare.desy.de/index.php/s/rSysTmdGAtYAf7r)

The folder has a size of approximately 5.6 GB and contains:
* Three operators that implement the Radon transform for inputs of different dimensions, created with the  [`radon` package](https://github.com/AlexanderAuras/radon), see also [[1]](#1). To simplify the computations we perform in our experiments, each operator contains the matrices $U,V$ and the vector $\sigma$ that form the singular value decomposition 
$$A = V\mathrm{diag}(\sigma)U^{T}.$$
The pre-saved operators have the following specifications:
    - `radon_32.obj` (21.6 MB) with forward operator $A: \mathbb{R}^{32\times 32} \rightarrow \mathbb{R}^{32\times 47}$
    - `radon_64.obj` (343 MB) with forward operator $A: \mathbb{R}^{64\times 64} \rightarrow \mathbb{R}^{64\times 93}$
    - `radon_128.obj` (5.3 GB) with forward operator $A: \mathbb{R}^{128\times 128} \rightarrow \mathbb{R}^{128\times 183}$

> [!TIP] 
> If you would like to create a radon operator of another dimension, we recommend using `radon_matrix` from the [`radon` package](https://github.com/AlexanderAuras/radon). It returns a matrix which can be passed directly to the class `ops.svd_op` we provide in this repository. A minimum working example would look like this:
  > ```
  > from radon import radon_matrix 
  > from ops import svd_op
  > import torch
  >
  > res = 16                     # your favorite resolution
  >
  > A = radon_matrix(img = torch.zeros((res,res)), 
  >                 thetas = torch.linspace(0.0, torch.pi, res)[:-1])
  >
  > op16 = svd_op(A,res)     # computes the SVD and implements forward, inverse and adjoint for 2-D inputs
  >```

* Three lists (`pi32.txt`, `pi64.txt`, `pi128.txt`) of coefficients $\{\Pi_n^\*\}_{n \in \mathbb{N}}$ corresponding to the three pre-saved operators. The data distribution $\pi^\*$ is described by the first 2400 images of the [LoDoPaB CT data set](https://zenodo.org/records/3384092), see also [[2]](#2).\
  The code we have used to compute the coefficients is documented in `coeffs.py` and can be applied to any other combination of data and operator.
* An example image `test_img.png`, taken from the [LoDoPaB CT data set](https://zenodo.org/records/3384092).

### <a id="5"> Requirements </a>
Using the pre-saved data, running the code only requires the packages `matplotlib`, `numpy`, and `scikit-image`. 
You can install these packages by running
```
pip install -r requirements.txt
```
in the project directory.

### <a id="6"> Experiments </a>
The aim of the experiments we provide here, is to compare different data-driven spectral reconstruction operators to solve inverse problems. More precisely, for an inverse problem $y= Ax + \epsilon$, we consider reconstruction operators 

$$ R_{\mu}(y) = \sum_{n = 1}^N g_n(\mu)  \langle y, v_n \rangle  u_n = U\mathrm{diag}(g(\mu))V^T  y.$$ 

Here, $A = V\mathrm{diag}(\sigma)U^{T}$ is the [singular value decompositon](https://en.wikipedia.org/wiki/Singular_value_decomposition) of $A$. The n-th columns of $U$ and $V$ are denoted by $u_n$ and $v_n$.

The filter $g(\mu) = \{g_n(\mu)\}_{n \in \mathbb{N}}$ is now optimized with respect to three different learning paradigms (which we call `mse`, `adv` and `post` in the code), \
with training noise $\epsilon$ drawn from the distribution $\mu$ and data $x$ drawn from the distribution $\pi^\*$.\
For all considered approaches, the optimal filter coefficients are of the form 

$$ g_n(\mu) = \frac{\Pi^\*_n\sigma_n}{\Pi_n^\*\sigma_n^2 + \lambda_n(\sigma_n, \Pi^\*_n, \Delta_n(\mu))}.$$

Here, $\sigma_n$ denotes again the singular values, and the data-driven, or, respectively, noise-driven coefficients are given by

$$ \Pi_n^\* = \mathbb{E}\_{x \sim \pi^\*}[\langle x, u_n\rangle^2] \quad \text{and}\quad \Delta_n(\mu) = \mathbb{E}\_{\epsilon \sim \mu}[\langle \epsilon, v_n\rangle^2].$$

The specific coefficients for the different approaches are then given by

$$ \color{darkorange}{\lambda_n^{\text{mse}}(\mu)} = \Delta_n(\mu),\qquad \color{teal}{\lambda_n^{\text{adv},3/8}(\mu)}= \frac{\Delta_n(\mu)}{3\sigma_n^2\Pi_n^\* + \Delta_n(\mu)}, \qquad \color{red}{\lambda_n^{\text{post}(\mu)}} = \sigma_n^2 \Delta_n(\mu).$$


Our driving questions are:
1. Are the data-driven reconstruction operators actually regularizers, i.e., stable?
2. If yes: Is the regularization convergent, i.e., do we converge to a "ground truth"-reconstruction as the noise level tends to zero?
3. If yes: Is the regularization still convergent if we test it on noise drawn from $\nu \neq \mu$?

In the experiments, we perform CT-reconstruction, where the forward operator $A$ is given by the [Radon transform](https://en.wikipedia.org/wiki/Radon_transform).

The main parts of this repository are the two Jupyter Notebooks `continuity.ipynb` and `convergence.ipynb`.
* In `continuity.ipynb` (which complements Section 6.1 of our paper) we document the experiments on the reconstruction error for a *fixed and known noise model* on three *different image resolutions*.
* In `convergence.ipynb` (which complements Section 6.2 of our paper) we document the experiments on the reconstruction error for a *fixed image resolution* but three *different noise models and decaying noise level*.

> [!TIP]
> The object `radon_128.obj` is quite large and computations using the `128x128`-operator can take quite some time. 
> If you have problems downloading the pre-saved operator or it takes too long to run the Code on your machine, you can skip the parts that use `res = 128`, or, respectively, change them to `res = 64` or `res = 32`.

### <a id ="7"> References </a>
<a id="1">[1]</a> 
Kabri, S., Auras, A., Riccio, D., Bauermeister, H., Benning, M., Moeller, M., Burger, M.
Convergent Data-Driven Regularizations for CT Reconstruction. Commun. Appl. Math. Comput. (2024). \
<a id="2">[2]</a> 
Leuschner, J., Schmidt, M., Otero Baguer, D., Maass, P. 
LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction.
Scientific Data, 8(109). (2021).

### <a id ="8"> Citation </a>
If you would like to cite our preprint, you can copy the following bib entry:
```
@misc{burger2023learned,
      title={Learned Regularization for Inverse Problems: Insights from a Spectral Model}, 
      author={Martin Burger and Samira Kabri},
      year={2023},
      eprint={2312.09845},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```



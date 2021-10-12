# Input Encoding

As it is common with many ML tasks, before fitting the model, we need to transform the data in some way or another. In Reinforcement Learning, this tasks is known as encoding and can be completely custom to your particular task.

:::{note}
A simple example of data encoding would be the use of Convolutional Neural Networks to process the raw pixel data from images when we want to train Reinforcement Learning algorihtms on video games for instance.
:::

The authors of the paper used their own graph convolution-based encoder which takes the board or order representations described earlier and transforms them using:
* Graph Convolutional Layer 
* BatchNormalization 
* FiLM 
* ReLU

At the end we are left with final provice embeddings which will be used to get outputs.

Let's take a look at the encoding process with the board state. Previous phase orders are encoded in the exact same manner.



## Encoding process

Graph based encoder was chosen in order to take advanteage of the adjacency information of the game map. Since graph neural networks work with data relationship by design and many of the GNN algorithms use neighbouring information for the encoding process, using graph based encoder seems to make sense.

To begin with, let's establish some notation:  
$  x_{b o}^{l}  $ - is the board state embedding produced by layer *l*  
$  x_{p o}^{l}  $ - is the previous orders embedding produced by layer *l*  

:::{note}
The layer here refers to a encoding block, which as described earlier consits of:
* Graph Convolutional Layer 
* BatchNormalization 
* FiLM 
* ReLU

Thus we can reuse this block and run the board state or previous orders embeddings multiple times through it.
The original paper did 16 such loops!
:::

Now before everything, our $  x_{b o}^{0}  $ and $  x_{p o}^{0}  $ will of course be our original board state and previous orders representations, which we disscussed in the previous page. Let's continue with the board state only for now (remeber, the process is exact for the orders).

We start our encoding by aggregating neighbour information, which is a key trait of most GNN:  
$
y_{b o}^{l}=\operatorname{Batch} \operatorname{Norm}\left(A x_{b o}^{l} W_{b o}+b_{b o}\right)
$

Here:  
$ A $ - is a normalized map 81x81 adjeceny matrix (the map has 81 provinces)  
$ W_{b o} $ - is the weight matrix  
$ b_{b o} $ - is the bias term

```
The *BathNorm* is operated on the last dimension (??????????)
```

Now, we perform additional conditional batch normalization using FiLM. According to the authors, it was used in order to fuse multimodal information, which FiLM layers excel at. The batch normalization parameters $ \gamma_{b o} $ and $ \beta_{b o} $, which allow the model to choose optimal distributions for hidden layers are conditioned on the player's power **p** and the current season **s**.  
$
\gamma_{b o}, \beta_{b o}=f_{b o}^{l}([p ; s]) 
$  
Here $ f^{l} $ is a linear transformation.

The FiLM works simply by applying the simple affine transformation:  
$
z_{b o}^{l}=y_{b o}^{l} \odot \gamma_{b o}+\beta_{b o}
$  
Both Hadamard product (element-wise multiplication) and additional are broadcast across provinces.

Finally, we add non-linear transformation in a form of ReLU with residual connections:  
$
x_{b o}^{l+1}= \begin{cases}\operatorname{Re} L U\left(z_{b o}^{l}\right)+x_{b o}^{l} & d^{l}=d_{b o}^{l+1} \\ \operatorname{Re} L U\left(z_{b o}^{l}\right) & d_{b o}^{l} \neq d_{b o}^{l+1}\end{cases}
$

```
I don't understand what d is, it's never explained, or maybe I am blind!
```

And that's it, we produced the board state embedding!  
Again, this process can now be repeated to get more representative embeddings.  

Finally, after running the $ x_{b o} $ and $ x_{p o} $ through L (just a number) encoding blocks, we concatenate them together to form the final embeddings:  
$
h_{e n c}=\left[x_{b o}^{L}, x_{p o}^{L}\right]
$

So now, $ h_{e n c}^{i} $ is a final embeding for the *i'th* province.












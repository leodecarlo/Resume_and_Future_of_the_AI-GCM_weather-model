# Optmizations and possible Evolutions of the _UNet_like_Generator_ Architecture

In the section [Optimization](#optimization) we present some possible improvements to the current Architecture, basically following _Neural-Networks-training-good-practices_. We will refer to the Chapters and Lectures of [FF_Deep_Learning](https://fleuret.org/dlc/), for more specific references see there. In [Deep Learning Course](https://fleuret.org/dlc/) you find slides (which are the material of the videos) and handouts which are extended slides with more explanations.  In this folder [my-FF_DL_course](https://cineca.sharepoint.com/:f:/r/sites/HPC/Documenti%20condivisi/Projects/Funded/TheAILAM/PaperUtili/Deep_Learning_Course_FFLeuret?csf=1&web=1&e=uklrLe) you find all the material (excluded videos) to be downloaded in only one time, where there are also the files _all_handouts_X.pdf_ containing in one .pdf all the Lectures X.x of the Chapter X (with some extra notes that I added).

In the section [Evolutions](#evolutions) we will present some proposals about how our Architecture could  evolve, in particular how it can evolve into the spirit of a dynamical system with  initial boundary conditions $X(n+1)=F(X(n)), X(0)=X_0$, i.e. given an initial input $X_0$ we want to predict the next steps (for us the next lead times that we want as weather predictions).<br>
 **<u>REMARK</u>**: Enforcing the system to predicts variuous lead times, i.e. giving a time dimension to the model, could fix the delay of the model in forecasting "Atmospheric Circulation Changes".  The level of the time representation can be taught at different levels of architecture complexity.


 ## Remarks about current model [UNetIllumia_torch_1.py](Pytorch_porting_of_UNet-Illumia/UNetIllumia_torch_1.py)

0. ***Avoid to risk to divide by zero***:  if one divides by std in the preprocessing normalization, it is appropiated to divide by $\sqrt{\text{variance} + \varepsilon}$  for some small $\varepsilon>0$ (i.e. avoid to divide by  something too close to zero).

1. ***Leaky-RELU and RELU***: Initially our data are all between 0 and 1, after matrix multiplications they can become negative. We observe that physically our quantities are non-negative; therefore, after matrix multiplications, Leaky-RELU, which allows negative values, could propagate them. Therefore, it could be more appropriate to use the simpler RELU as an activation function. This, plus the [Pixel Normalization](#pixel-normalization) (described below), would implicitly maintain the data between 0 and 1 (as the initial input).



## Optimization

List of improvements for good training practices in possible order of importance:


1. ***Parameter Initialization***: the design of the parameters initialization aims at controlling the variance of the gradients so that weights evolve at the same rate across layers during training, and no layer reaches a saturation behavior before others. As the Network becomes deeper an issues is the stability of the gradients and introducing scaling factors is strongly suggested, *"Xavier initialization"* is the common suggestion (different activation function could need sligtly different scaling factors). For this point see [Lect. 5.5](https://cineca.sharepoint.com/:f:/r/sites/HPC/Documenti%20condivisi/Projects/Funded/TheAILAM/PaperUtili/Deep_Learning_Course_FFLeuret/5?csf=1&web=1&e=omt5Iw).  See the effect across layers at page 16.

2.  ***$L¹/L²-$ regularization***: possible overfitting could be reduced imposing, in addition to the  statistical loss on the data, a loss on the weight. This is called "_regularization_" , the common choices are $L¹$ and $L²$ _regularizations_ or both, to make the model less dependent on the data. The choice depends   on the policy you want on "_sparsity_" in the model parameters. See [Lect. 5.4](https://cineca.sharepoint.com/:f:/r/sites/HPC/Documenti%20condivisi/Projects/Funded/TheAILAM/PaperUtili/Deep_Learning_Course_FFLeuret/5?csf=1&web=1&e=cFGsGz).

3. ***Training-normalization***:  During training the statistics/scale of layers shifts, the motivation for normalizing the features of the layers, during the training,   is that if the statistics/scale of the activations are not controlled during training, a layer will have to adapt to the changes of the activations computed by the previous layers in addition to making changes to its own output to reduce the loss [Lect. 6.4](https://cineca.sharepoint.com/:f:/r/sites/HPC/Documenti%20condivisi/Projects/Funded/TheAILAM/PaperUtili/Deep_Learning_Course_FFLeuret/6?csf=1&web=1&e=12tYb2). The most common *Batch-Normalization* is not suitable for our case because we average across all the space dimensions, which is of course not what we want (_mutatis mutandis_ for *Layer,Instance and Group Normalizations* ).<br>
What we want  is coherent statistics of layers/scale across epochs, two options that could fit into our case:<br>
   - ***Pixel normalization***:  normalize by $L^2$ norm: for each sample $n$ and each spatial location $(h, w)$, you sum on channels $C$ : $\text{norm}_{2,n,h,w} =\sqrt{\frac{1}{C}\sum_{c=1}^{C} x_{n,c,h,w}^2 +\varepsilon}$,
      then divide the channel vector by that norm: $\frac{x_{n,c,h,w}}{\text{norm}_{2,n,h,w}}\rightarrow x'_{n,c,h,w}$. The factor $1/C$ is just formal, it does not have any real role, hence I would  remove it to have $\|x'_{n,c,h,w}\|_2= \|x'_{n,c,h,w}\|_2^2=1$, combing this choice with RELU would guarantee that the component $(x'_{n,c,h,w})_i$ are between 0 and 1 at every layer, as our initial preprocessing normalization. Also, what is of our interest is avoiding to touch the spatial dimension, so we could also normalize on minibatch, i.e. a normalization with $\text{norm}_{2,h,w} = \sqrt{\frac{1}{C\times N} \sum_{n=1}^{N}\sum_{c=1}^{C} x_{n,c,h,w}^2 + \varepsilon}$.<br>
      This type of normalization just ensures all channels at a given spatial location have a controlled magnitude and avoid “feature scale explosion” across channels at each pixel location. Pixel normalization is not natively implemented in PyTorch, but there are around implementations by users.<br>

   - ***Weight Normalization and Spectral Normalization***:  here, at each layer, weights are reparametrized as  $\mathbf{w} = g \frac{\mathbf{v}}{\|\mathbf{v}\|}$, where $g$ is a scalar (learned parameter), and  $\mathbf{v}$ is a vector (also learned) of the same shape as $\mathbf{w}$. So this normalization aims to stabilize the training at the level of weights. It is natively implemented in PyTorch [native PyTorch Weight Normalization implementation](https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html).<br>
   Another normalization on weights instead than on features is ***Spectral Normalizations***, this is a layer normalization that estimates the largest singular value of a weight matrix, and rescale it accordingly, see page 15 and 16 of [Lect. 11.2](https://cineca.sharepoint.com/:f:/r/sites/HPC/Documenti%20condivisi/Projects/Funded/TheAILAM/PaperUtili/Deep_Learning_Course_FFLeuret/11?csf=1&web=1&e=CFYwZ5). This normalization is also natively implemented in PyTorch [Native PyTorch implemenation of Spectral Normalization](https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html).


   I don't fully grasp the meaning of this type of normalizations on weights, since the original motivation for normalizations in layer during training is that the scales of features may fluctuates a lot, making difficult the convergence of the Networks. Nevertheless they are stated to stabilize(and help the gradient flow) the training in Networks similar to our Generator [UNetIllumia_torch_1.py](Pytorch_porting_of_UNet-Illumia/UNetIllumia_torch_1.py).


4. ***Dropout***: Dropout is another alternative regularization techniques, which is intended to promote generalization. The idea is to train an ensembles of models with heavy weight sharing. Standard [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) of Pytorch is not a suggested choice for our case, since we remove spatial information which is very important for us. An alternative for activation maps that are generally locally correlated is  [nn.Dropout2d](https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html). This dropouts entire channels, having at the initial layer different physical quantity it does not seem a good idea to apply it at least at the beginning but it could be ok to apply Dropout2d at deeper layers (where the channels represent learned features). Finally the third option(that seems the best fit for weather data) is to implement dropconnect, that is dropping connection instead than features. This is called [Dropconnect](https://onedrive.live.com/?cid=889319B7F408E8A1&id=889319B7F408E8A1%2166795&parId=889319B7F408E8A1%2156047&o=OneUp) is not already implemented in PyTorch, it is computationally intensive and request enough work to be implemented. See pages 10-11 there. For all the topic of Dropout you can refer to [Lect 6.3](https://cineca.sharepoint.com/:f:/r/sites/HPC/Documenti%20condivisi/Projects/Funded/TheAILAM/PaperUtili/Deep_Learning_Course_FFLeuret/6?csf=1&web=1&e=t89sfA).


**<u>BOTTOM LINE</u>**: a reasonable way to proceed would be implementing [Xavier Initialization Normalization](#parameter-initialization) and $L¹/L²-$ regularization. Later if there is time, [Pixel Normalization](#pixel-normalization)  and [Dropconnect](#dropout). 

## Evolutions


- One of our major discussion is about the laziness of the model, i.e  the model in general forecasts what will happen but with some delay. One possible solution is to give it a time dimension  designing an encoder that shares its bottleneck (latent dimension) middle layer with  different decoders(one for each lead time, plus eventually one to reconstruct the initial $X(0)$). So to enforce the model to predict an entire trajectory, i.e. an evolution of the system.  Each Decoder $i$ has its loss $\mathcal{L}_{Decoder(i)}$ but the training optimizes the entire network with a total loss $\sum_i \alpha(i)\mathcal{L}_{Decoder(i)}$, where the sum is over all the decoders and $0<\alpha(i)<1$ such that $\sum_i \alpha(i)=1$.<br>
The simplest implementation is given by  separated parallel decoders, optimizing the sum of the losses. The  different enconders proceed separately and they are affected by each other just trough the constraints imposed by the joint optimization of the total loss function. Here  a scheme with two decoder:


```
encoder = SharedEncoder()
decoder_one = DecoderOne()
decoder_two = DecoderTwo()

# Forward pass
latent = encoder(input_data)
output_one = decoder_one(latent)
output_two = decoder_two(latent)

# Compute losses and backpropagate
loss_one = criterion(output_one, target_one)
loss_two = criterion(output_two, target_two)
total_loss = alpha * loss_one + beta * loss_two
total_loss.backward()
optimizer.step()
```

 In this approach all lead times are treated equally, while one could want that there is an explicit time direction between different decoders, since physically what happens at a given time is caused by what happened in the past. A way to address this issue is to add some attention mechanism for the decoder $D_{T}$  at lead time $T$ with respect  to layer of the decoder $D_{T-1}$ at lead time $T-1$, for example with some unidirectional connection from the layers of $D_{T-1}$ to the ones of $D_{T}$. Anyway the general rule is to stay simple as much as possible and it is also true that the solution of a physical system is $X'= f(X)$ is expected to be unique, therefore the state of the system $X_{T-1}$ is unique given $X_{T}$ (i.e backward uniqueness).
Eventually a simpler way to give a hierarchy between decoders is using a $\alpha(i)$ strictly decreasing, i.e. $\alpha(i-1)< \alpha(i)$, that is making weaker the optimization with respect to greater lead times (which could make sense since, due to the chaotic nature of weather, forcasting becomes more and more uncertain during time).

About this approach, we should remember this fact on optimization: $\min_x (f(x)+g(x)) \geq \min_x f(x) + \min_x g(x)$, i.e. the min of the sum is in general greater than the sum of the min. Therefore the decoders we will get could be suboptimal on the training set, if compared with respect to the decoders trained singularly. On the other side, the joint training could promote generalization.



# Notes

This file contains additional notes, including demonstration hyperparameter sets, for each of the notebooks in the repository. These are kept separate so the notebooks can be encountered "spoiler-free" first!

### Table of contents
* [f_s: `fit_sinewave`](#f_s-fit_sinewave)
  * [f_s.1: One node for each input](#f_s1-one-node-for-each-input)
  * [f_s.2: Simpler model](#f_s2-simpler-model)
  * [f_s.3: Deep model](#f_s3-deep-model)
  * [f_s.4: Implications](#f_s4-implications)
* [nn_c_pp: `nn_class_palmerpenguins`](#nn_c_pp-nn_class_palmerpenguins)
  * [nn_c_pp.1: Simple model](#nn_c_pp1-simple-model)
  * [nn_c_pp.2: Autoencoder](#nn_c_pp2-autoencoder)
  * [nn_c_pp.3: Implications](#nn_c_pp3-implications)
* [a_o_j: `ann_ozone_joshuatree`](#a_o_j-ann_ozone_joshuatree)
  * [a_o_j.1: One possible model](#a_o_j1-one-possible-model)

## f_s: fit_sinewave
This problem is deceptively challenging! It is not trivial to represent periodic behavior purely through statistics.

### f_s.1: One node for each input
This gives one node for each input into the first hidden layer. This is probably overkill--but it definitely gets the job done!  

The training dynamics for this model reveal an interesting phenomenon: there is a sharp decrease in loss in the first ~25 epochs, then a plateau until epoch ~60, then an even sharper decrease between epoch 60 and 70 until the loss is almost zero. This shows the importance of training for a sufficient number of epochs. There are usually multiple local minima in a loss function; the model may need a little time to find its way to an adequate minimum.

* `learning_rate = 0.01`
* `batch_size = 2`
* `n_epochs = 100`
* `activation = 'sigmoid'`
* `hiddens = [1000,]`
* `loss = 'mse'`

Try increasing the `learning_rate` to 0.1 and leaving the other hyperparameters the same. The model converges to its "optimal" solution much faster--but the solution is not as good!

### f_s.2: Simpler model
A sine wave can be approximated with a relatively simple ANN.  

* `learning_rate = 0.1`
* `batch_size = 2`
* `n_epochs = 25`
* `activation = 'sigmoid'`
* `hiddens = [10,]`
* `loss = 'mse'`

As there is only one hidden layer neither this example nor the one in f_s.1 are "deep learning" models.

### f_s.3: Deep model
The sine wave can also be approximated with a more complex deep neural network.

* `learning_rate = 0.01`
* `batch_size = 5`
* `n_epochs = 100`
* `activation = 'sigmoid'`
* `hiddens = [5, 8]` 
* `loss = 'mse'`

Note from the training dynamics that this displays a similar local minimum as the model in **f_s.1**.  

*Model selection contributed by [Herijaona Hani-Roge Hundilida Randriatsara](https://www.linkedin.com/in/hundi-randriatsara/).*

### f_s.4: Implications
This example is idealized--it's rare you need to directly approximate a sine wave! But, the different models illustrate a key principle: many different model architectures can give similar results. In the mathematical theory of model development, this is called the *Rashomon Effect* (e.g., Rudin 2024 [link](https://arxiv.org/pdf/2507.03884)).  

The Rashomon Effect raises many questions about model development, for example:
* Should we train a deep model if a simpler one is adequate for purpose?
* If we understand the dynamics of a system (in this case, `sin(x)`), should we apply that dynamical knowledge, these statistical methods, or both? 
* Is predictive accuracy the primary metric of model adequacy, or are other principles important (e.g., interpretability, explainability, efficiency)? What if there are tradeoffs between multiple principles that we care about?

## nn_c_pp: nn_class_palmerpenguins

### nn_c_pp.1: Simple model
The penguins can be classified with a simple model consisting of a single layer with 10 nodes.
* `loss: tf.keras.losses.CategoricalCrossentropy(from_logits=True)`
* `"metric": 'accuracy'`
* `"hidden_nodes": [10,]`
* `"out_nodes": 3`
* `"rnd_state": 13`
* `"activations": {"hid": 'relu', "out": 'softmax'}`
* `"num_epochs": 50`
* `"batch_size": 32`
* `"learn_rate": 0.01`
* `"initializer": tf.keras.initializers.RandomNormal`
* `"regularizer": None`
* `"early_stop": tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='auto', restore_best_weights=True)`
* `"verbosity": 1`
* `"class_weight": None`

### nn_c_pp.2: More complex model
The penguins can also be classified with a much more complex model. This architecture is similar to an *encoder-decoder*, which refers to a structure where the data is squashed down to the smallest number of nodes possible in an interim layer ("latent space"; in this case, the 5-node layer) then expanded again. (Technically, an encoder-decoder should have inputs and output of the same length.) This can act as a way force the network to focus on the most important features. It can also be an effective method of compressing data!
* `loss: tf.keras.losses.CategoricalCrossentropy(from_logits=True)`
* `"metric": 'accuracy'`
* `"hidden_nodes": [15, 5, 15]`
* `"out_nodes": 3`
* `"rnd_state": 13`
* `"activations": {"hid": 'relu', "out": 'softmax'}`
* `"num_epochs": 50`
* `"batch_size": 32`
* `"learn_rate": 0.01`
* `"initializer": tf.keras.initializers.RandomNormal`
* `"regularizer": None`
* `"early_stop": tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='auto', restore_best_weights=True)`
* `"verbosity": 1`
* `"class_weight": None`

Experiment with adjusting the size of the latent space. You cannot make the latent space any smaller without losing critical information for classification and compromising model performance.

### nn_c_pp.3: Implications
In my experience, it is actually fairly straightforward to find a network that almost perfectly classifies our penguins! However, nothing tells you that you've reached a stopping point. You have to exercise your own judgment as a scientist to decide that any given model is good enough. We can think of this in terms of two principles:  

* *Don’t be afraid to try many options!* Iterative refinement is key to making a successful network. 
* *Don't be afraid to stop!* Remember that you are creating a model for a specific purpose to address a specific question. Once you have obtained a model that is adequate for your given purpose, you can stop there! It is not our goal to obtain the "perfect model." (Perfection is not even a well-defined goal for most problems!)

## a_o_j: ann_ozone_joshuatree

### a_o_j.1: One possible model
This produces a model with the following metrics: 
* Mean absolute error: 0.46595
* Median absolute error: 0.34917
* Root mean squared error: 0.62492
* Maximum error: 2.93097
* Variance explained: 59.984%
  
With this set of input features (temperature, relative humidity, wind speed, day of year), I haven't been able to improve performance significantly beyond this! Please let me know if you find an approach that performs dramatically differently from this and I'll add it here.

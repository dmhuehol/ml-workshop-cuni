# Notes

Notes on tuning, data preparation, and other factors for each of the notebooks demonstrated here. These are kept separate so the notebooks can be encountered "spoiler-free" before seeing the notes here!

## f_s: fit_sinewave
This problem is deceptively challenging! It is actually not trivial to represent periodic behavior in a purely statistical framework.

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

As we only have one hidden layer this is not a "deep learning" model.

### f_s.3: Deep model
The sine wave can also be approximated with a more complex deep neural network.

* `learning_rate = 0.01`
* `batch_size = 5`
* `n_epochs = 100`
* `activation = 'sigmoid'`
* `hiddens = [5,8]` 
* `loss = 'mse'`

Note from the training dynamics that this displays a similar local minimum as the model in **f_s.1**.  

This example is idealized--it's rare you need to directly approximate a sine wave! But, this illustrates a key principle: many different model architectures can give similar results. In the mathematical theory of model development, this is called the *Rashomon Effect* (e.g., Rudin 2024 [link](https://arxiv.org/pdf/2507.03884)).  

The Rashomon Effect gives rise to many questions about model development, for example:
* Should we train a deep model if a simpler one is adequate for a purpose?
* If we understand the dynamics of a system (in this case, `sin(x)`), should we apply that dynamical knowledge, these statistical methods, or both? 
* Is predictive accuracy the primary metric of model adequacy, or are there other principles that guide us (e.g., interpretability, explainability, efficiency)? What if there are tradeoffs between principles that we care about?

These questions are directly relevant in model applications--for example, in how models are evaluated and adopted in higher-stakes environments like operational forecasting.  
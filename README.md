This is the TensorFlow implementation based on the paper, "[Hierarchical Attention Networks for Document Classification](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)" .

There are a few issues at the moment,

* It does not support mini batch at the moment. There is an issue regarding nested `while_loop` opened to track root cause of the [issue](https://github.com/shengc/tf-han/issues/1). 
* The notebook provides an example of applying `HAN` to learn from 20 news group dataset. The performance on testing data set is pretty bad. The reason could be the model was not trained on the entirety of the training set. What's even more intriguing is using the GloVe word embeddings out of the box actually produces very bad results, even on training data set.

Gotcha,

* The original paper does not use `cross-entropy` as the Loss function, instead it uses the negative sum of the logarithm of the probability corresponding to the correct class label.  

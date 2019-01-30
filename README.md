# CNN-Autoencoder
A convolutional autoencoder neural network for vegetal electrophysiological classification

get_actions.py contains net architecture and weights, and script handling. AE_Test.py shows usage.

action clustering

As far as the axes you are talking about, they are “abstract” features derived from a NN processing your data, from which the data can be clustered to different actions

 I can provide a script that will abstract everything for you and you’ll only call a function in it every time you get a new sample and it’ll return the current action
 
 
![t-SNE clustering](CNN-Autoencoder/TSNE Clustering.png?raw=true "Title")
      

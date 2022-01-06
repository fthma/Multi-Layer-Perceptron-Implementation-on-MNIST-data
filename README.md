# Multi-Layer-Perceptron-Implementation-on-MNIST-data

This project implements a Multi- Layer Perceptron in recognising Handwritten Digit Images into 
Digits from (0-9). The Handwritten digit images are taken from the MNIST dataset . Two sets of 
image datasets are taken, the smaller one is reserved for testing and is used only after the model is 
built using the other dataset. The multi-layer perceptron model is built by taking 10000 images for 
training the model and 3000 images from the testing dataset. Then the Error is calculated by 
comparing the output with the corresponding labels data. We make use of the Logistic sigmoid for 
activation function in this project to improve the learning of the model . The model is built multiple 
times with 3 different values of learning rates with different value so hidden nodes=10,35,100. The 
percentage error in recognising each digit is found for all three models constructed with three 3 
different values of learning rates and it is plotted on bar charts.

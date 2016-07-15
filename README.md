# TF-HPO
TensorFlow Deep Learning Hyperparameter Optimization

----
This project aims to find the optimal way to select the hyperparameters of Deep Neural Net (DNN) in TensorFlow. 



  * __Design Goal__
  
    I design this project to find the following research questions:
     * How many neurons will be optimal to get the best accuracy?
     * How many layers will be fitted?
     * Which architectures will be adaquate for the famous dataset such as MNIST and so on.

  * __Test Environments__
  
    This project is being developed under the following environments:
      * jupyter notebook on Ubuntu 14.04 with a high performance PC (Intel Core i7, 16G RAM, Single GeForce GTX 1080)
      * jupyter notebook over docker on Windows 8.1 with general performance PC (Intel Core i7, 8G RAM, No GPU support)  
 
 * __System Configuration__
 
    This project is consisted of two parts:
      * _TrainManager.ipynb_ : control the trainings by the magic command to prevent MemoryExhaustionError
      * _{DNN architecture}_layer{number}.py_ : Actual python code to train a specific DNN. It requires more arguments to proper execution. See the help (-h) for your information.
      

  * __Key Considerations__

    * TensorFlow latest version to test (r0.9 working) may leak the resources and it will stop after a few iterations. This project designed to keep going to train after this unwantted stop for the long run.  

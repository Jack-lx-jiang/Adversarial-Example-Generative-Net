# Adversarial-Example-Generative-Net
A generative network that can produce adversarial examples quickly.

# Dependency
1. Tensorflow
2. cleverhans https://github.com/tensorflow/cleverhans

# Train procedure
1. Train target model. Run mdt_cifar10.py
2. Train AEGN. Run adv_encoder_target_exp.py

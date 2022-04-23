import torch
from torch.autograd.grad_mode import F
import torch.nn.functional as Fun
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    loss = None
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    assert logits_fake.shape == logits_real.shape
    N = logits_real.shape[0]
    loss_real = bce_loss(logits_real, torch.ones(logits_real.shape).to(device), reduction='mean') / 2
    loss_fake = bce_loss(logits_fake, torch.zeros(logits_fake.shape).to(device), reduction='mean') / 2
    ##########       END      ##########

    return loss_real + loss_fake


def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss = bce_loss(logits_fake, torch.ones(logits_fake.shape).to(device), reduction='mean')

    ##########       END      ##########

    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################
    real_loss = Fun.mse_loss(scores_real, torch.ones(scores_real.shape).to(device), reduction='mean') / 2
    fake_loss = Fun.mse_loss(scores_fake, torch.zeros(scores_fake.shape).to(device), reduction='mean') / 2
    ##########       END      ##########

    return real_loss + fake_loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss = Fun.mse_loss(scores_fake, torch.ones(scores_fake.shape).to(device), reduction='mean')
    ##########       END      ##########

    return loss

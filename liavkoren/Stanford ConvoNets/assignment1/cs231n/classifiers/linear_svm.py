import numpy as np
# from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (Data, Classes) containing weights.
  - X: A numpy array of shape (N, Data) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < Classes.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  N_train = X.shape[0]
  loss = 0.0
  # import pdb; pdb.set_trace()
  for data_index in xrange(N_train):
    scores = X[data_index].dot(W)
    correct_class = y[data_index]
    correct_class_score = scores[correct_class]
    for class_index in xrange(num_classes):
      if class_index == correct_class:
        incorrect_class_scores = np.hstack((scores[:class_index], scores[class_index+1:]))
        dW[:, class_index] += -1 * np.sum((incorrect_class_scores - correct_class_score) > -1 ) * X[data_index]
      else:
        score_diff = scores[class_index] - scores[correct_class] + 1
        dW[:, class_index] += np.sum(score_diff > 0) * X[data_index]
      margin = scores[class_index] - correct_class_score + 1 # note delta = 1
      # print(f'[{data_index:.3f}, {class_index:.3f}]: {scores[class_index]:.3f} - {correct_class_score:.3f} + 1 = {margin:.3f}')
      if margin > 0:
        loss += margin
        # print(f'  {loss:.3f}')

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by N_train.
  loss /= N_train
  dW /= N_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.

  Inputs:
  - W: A numpy array of shape (Data, Classes) containing weights.
  - X: A numpy array of shape (N, Data) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < Classes.
  - reg: (float) regularization strength

  """
  loss = 0.0
  dW = np.zeros(W.shape) # (Data, Classes)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)  # (N, Data) x (Data, Classes) = (N, Classes)
  N_train = X.shape[0]
  non_class_scores = np.copy(scores)
  non_class_scores[np.arange(N_train), y] = 0
  class_scores = np.sum(scores - non_class_scores, axis=1).reshape((N_train,1))

  # Question: This implementation puts 1.0s in all the correctly classified spots??!
  loss_matrix = scores - class_scores + 1
  loss_matrix = np.maximum(0, loss_matrix)

  loss = np.sum(loss_matrix)/float(N_train)
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dW = np.zeros(W.shape)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

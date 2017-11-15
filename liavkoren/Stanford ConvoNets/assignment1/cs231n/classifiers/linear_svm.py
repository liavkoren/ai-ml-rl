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
  dW = np.zeros(W.shape) # (D, C)
  # incorrect = np.zeros_like(W)
  # correct = np.zeros_like(W)
  # compute the loss and the gradient
  num_classes = W.shape[1]
  N_train = X.shape[0]
  loss = 0.0

  for data_index in xrange(N_train):
    scores = X[data_index].dot(W)  # -->(N, C) = (N, D) * (D, C)
    correct_class = y[data_index]
    correct_class_score = scores[correct_class]
    for class_index in xrange(num_classes):
      if class_index == correct_class:
        incorrect_class_scores = np.hstack((scores[:class_index], scores[class_index+1:]))
        dW[:, class_index] += -1 * np.sum((incorrect_class_scores - correct_class_score) > -1 ) * X[data_index]
        # correct[:, class_index] += -1 * np.sum((incorrect_class_scores - correct_class_score) > -1 ) * X[data_index]
      else:
        score_diff = scores[class_index] - scores[correct_class] + 1
        dW[:, class_index] += np.sum(score_diff > 0) * X[data_index]
        # incorrect[:, class_index] += np.sum(score_diff > 0) * X[data_index]
      margin = scores[class_index] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
  # dW = incorrect + correct

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by N_train.
  loss /= N_train
  dW /= N_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
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

  """
  We split the computation of the gradient into two parts: One branch computes the
  gradient for *incorrect* column: given an X_i and a y_i, we compute all the
  contributions to the gradient from the k-1 classifiers which are not y[i], and
  accumulate these contributions into a NxK matrix called `incorrect_dW`.

  Likewise, for each X_i and y_i pair, we accumulate the contribution of the y[i]
  correct classifier into a NxK matrix called `correct_dW`. The total gradient is
  the element wise sum of `incorrect` + `correct`.
  """
  incorrect_class_loss = scores - scores[range(N_train), y].reshape((N_train, 1)) + 1 > 0
  # null out the correct classifiers:
  incorrect_class_loss[range(N_train), y] = 0
  incorrect_dW = X.T.dot(incorrect_class_loss)

  # gives a T for each cell that contributes to the grad.
  correct_class_mask = (scores -  scores[range(N_train), y].reshape((N_train, 1))) + 1 > 0
  # null out the cells for the correct class
  correct_class_mask[range(N_train), y] = 0

  # collapse this into a vector that tells us how many classes are bad for each of the N rows in X:
  bad_classifiers_count = -1 * correct_class_mask.sum(axis=1)

  # now transform this into a mask that tells us how to manipulate X to get the grad:
  mask = np.zeros_like(scores)
  mask[range(N_train), y] = bad_classifiers_count
  correct_dW = X.T.dot(mask)

  dW = incorrect_dW + correct_dW
  dW /= N_train
  dW += reg * 2 * W
  return loss, dW

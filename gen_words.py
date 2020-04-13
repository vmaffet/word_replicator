#!/usr/bin/env python3

import numpy as np
import random

def clip(gradients, maxValue):
  for name,value in gradients.items():
    gradients[name] = np.clip(value, -maxValue, maxValue)
  return gradients


def softmax(x):
  e = np.exp(x - np.max(x))
  return e/np.sum(e,axis=0)


def initialize_parameters(n_a, n_x, n_y):
  Wax = np.random.randn(n_a, n_x)*0.01
  Waa = np.random.randn(n_a, n_a)*0.01
  Wya = np.random.randn(n_y, n_a)*0.01
  b   = np.random.randn(n_a, 1)
  by  = np.random.randn(n_y, 1)
  return {'Wax':Wax, 'Waa':Waa,'Wya':Wya, 'b':b, 'by':by}


def model(data, idx_to_chr, chr_to_idx, input_size, output_size, nb_ite=35000, n_a = 50, nb_examples = 7):

  # Retrieve n_x and n_y from vocab_size
  n_x, n_y = input_size, output_size
  params = initialize_parameters(n_a, n_x, n_y)

  # Initialize the hidden state of your LSTM
  a_prev = np.zeros((n_a, 1))

  np.random.shuffle(data)
  # Optimization loop
  for j in range(nb_ite):

    sample_idx = j % len(data)
    X = [None] + list(map(chr_to_idx.get, data[sample_idx]))
    Y = X[1:] + [chr_to_idx["\n"]]

    curr_loss, grads, a_prev = optimize(X, Y, a_prev, params)

    #loss = smooth(loss, curr_loss)

    if j % 2000 == 0:
      print(f'\nIteration: {j}, Loss: {curr_loss}')
      for name in range(nb_examples):
        sampled_indices = sample(params, chr_to_idx)
        word = translate_sample(sampled_indices, idx_to_chr)
        print(f'{word}',end='')

  return params


def rnn_forward(X, Y, a_prev, params):

  hidden_size, input_size = params['Wax'].shape

  x, a, y_pred = {}, {}, {}

  a[-1] = np.copy(a_prev)

  loss = 0
  for i,v in enumerate(X):

    x[i] = np.zeros((input_size, 1))
    if v is not None:
      x[i][v] = 1

    a[i], y_pred[i] = rnn_step_forward(params, a[i-1], x[i])

    loss -= np.log(y_pred[i][Y[i],0])

  cache = (y_pred, a, x)

  return loss, cache


def rnn_step_forward(params, a_prev, x):
  Waa, Wax, Wya, by, b = params['Waa'], params['Wax'], params['Wya'], params['by'], params['b']

  a_next = np.tanh(Wax @ x + Waa @ a_prev + b)
  z = Wya @ a_next + by
  y = softmax(z)

  return a_next, y


def rnn_backward(X, Y, params, cache):

  grads = {}

  y_pred, a, x = cache
  Waa, Wax, Wya, by, b = params['Waa'], params['Wax'], params['Wya'], params['by'], params['b']

  for p in params:
    grads[f'd{p}'] = np.zeros_like(params[p])
  grads['da_next'] = np.zeros_like(a[0])

  for i in reversed(range(len(X))):
    dy = np.copy(y_pred[i])
    dy[Y[i]] -= 1

    grads = rnn_step_backward(dy, grads, params, x[i], a[i], a[i-1])

  return grads, a


def rnn_step_backward(dy, grads, params, x, a, a_prev):

  da = params['Wya'].T @ dy + grads['da_next']
  da_raw = (1 - a*a) * da

  grads['dWya'] += dy @ a.T
  grads['dby']  += dy
  grads['db']   += da_raw
  grads['dWax'] += da_raw @ x.T
  grads['dWaa'] += da_raw @ a_prev.T
  grads['da_next'] = params['Waa'].T @ da_raw

  return grads


def update_params(params, grads, lr):

  for p in params:
    params[p] -= lr*grads[f'd{p}']

  return params


def optimize(X, Y, a_prev, params, lr=0.01):
  loss, cache = rnn_forward(X, Y, a_prev, params)

  grads, a = rnn_backward(X, Y, params, cache)

  grads = clip(grads, 5)

  params = update_params(params, grads, lr)

  return loss, grads, a[len(X)-1]


def sample(params, chr_to_idx, start=None, start_save=[]):
  if start is not None:
    for c in start:
       start_save.append(chr_to_idx[c])
    return

  Waa, Wax, Wya, by, b = params['Waa'], params['Wax'], params['Wya'], params['by'], params['b']
  visble_size = by.shape[0]
  hidden_size = Waa.shape[1]

  # First input (seed)
  x = np.zeros((visble_size, 1))
  # Hidden state init
  a_prev = np.zeros((hidden_size, 1))


  start_size = len(start_save)
  output = start_save.copy()
  sel_chr = -1

  eol_chr = chr_to_idx['\n']
  # Ensure we don't loop forever
  for i in range(50):

    if sel_chr == eol_chr:
      break

    # Go through network
    a, y = rnn_step_forward(params, a_prev, x)

    if i < start_size:
      sel_chr = output[i]

    else:
      # Pick next char at ramdom from network distribution
      sel_chr = np.random.choice(visble_size, p=y.ravel())
      output.append(sel_chr)

    # Change input according to choice and get ready for new iteration
    x *= 0
    x[sel_chr] = 1
    a_prev = a

  if output[-1] != eol_chr:
    output.append(eol_chr)

  return output


def translate_sample(sample, idx_to_chr):
  return ''.join(map(idx_to_chr.get, sample))


if __name__ == '__main__':

  data = set()
  chars = set()
  with open(input('File to learn from: '), 'r') as file:
    for line in file:
      data.add(line.lower())
      chars.update(set(line.lower()))

  data = list(data)

  nb_samples = len(data)
  nb_chars   = len(chars)

  print(f'{nb_samples} samples found\n{nb_chars} uniques chars')

  chr_to_idx = {c:i for i,c in enumerate(sorted(chars))}
  idx_to_chr = {i:c for c,i in chr_to_idx.items()}

  sample(None, chr_to_idx, start=input('Start with: ').lower())

  params = model(data, idx_to_chr, chr_to_idx, nb_chars, nb_chars, nb_ite=50000, n_a=50, nb_examples=4)

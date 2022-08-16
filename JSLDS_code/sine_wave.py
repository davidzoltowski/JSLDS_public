
"""Routines for creating white noise and integrated white noise."""

from __future__ import print_function, division, absolute_import
from functools import partial
import jax.numpy as np
from jax import jit, vmap
from jax import random
import jax
import matplotlib.pyplot as plt

def keygen(key, nkeys):
  """Generate randomness that JAX can use by splitting the JAX keys.
  Args:
    key : the random.PRNGKey for JAX
    nkeys : how many keys in key generator
  Returns:
    2-tuple (new key for further generators, key generator)
  """
  keys = random.split(key, nkeys+1)
  return keys[0], (k for k in keys[1:])

def build_input_and_target_sine_wave(input_params, key):
  """Build white noise input and integration targets."""
  minf, maxf, df, T, ntime = input_params
  dt = T/ntime

  # Generate random frequency 
  key, skey = random.split(key)
  maxval = int((maxf - minf) / df)
  freq = minf + df * random.randint(skey, shape=(), minval=0, maxval=maxval)
  inputs_tx1 = freq * np.ones((ntime, 1))
  targets = np.sin(freq * np.arange(ntime) * dt)
  targets_tx1 = np.expand_dims(targets, axis=1)
  targets_mask = np.expand_dims(np.arange(ntime), axis=1)
  return inputs_tx1, targets_tx1, targets_mask


# Now batch it and jit.
build_input_and_target = build_input_and_target_sine_wave


def build_inputs_and_targets(input_params, keys):
  f = partial(build_input_and_target, input_params)
  f_vmap = vmap(f, (0,))
  return f_vmap(keys)


def plot_batch(ntimesteps, input_bxtxu, target_bxtxo=None, output_bxtxo=None,
               errors_bxtxo=None, ntoplot=1):
  """Plot some frequency and target examples."""
  plt.figure(figsize=(10,7))
  plt.subplot(221)
  plt.plot(input_bxtxu[0:ntoplot,:,0].T, 'b')
  plt.xlim([0, ntimesteps-1])
  plt.ylabel('Frequency')

  plt.subplot(222)
  if output_bxtxo is not None:
    plt.plot(output_bxtxo[0:ntoplot,:,0].T);
    plt.xlim([0, ntimesteps-1]);
  if target_bxtxo is not None:
    plt.plot(target_bxtxo[0:ntoplot,:,0].T, '--');
    plt.xlim([0, ntimesteps-1]);
    plt.ylabel("Integration")
  if errors_bxtxo is not None:
    plt.subplot(224)
    plt.plot(errors_bxtxo[0:ntoplot,:,0].T, '--');
    plt.xlim([0, ntimesteps-1]);
    plt.ylabel("|Errors|")
  plt.xlabel('Timesteps')
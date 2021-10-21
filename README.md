# Real-Time DDSP Timbre Transfer in MATLAB

Requires MATLAB >= R2020b and the Audio Toolbox.

These plugins are based on Google Magenta's Differentiable Digital Signal Processing (https://github.com/magenta/ddsp).

DDSP trains an autoencoder to extract pitch, loudness and timbre information from a given audio signal, and to generate synthesizer parameters from this information to reconstruct the original audio.
If the autoencoder is trained on e.g. a violin sound, we feed its decoder with pitch and loudness information from any arbitrary sound source to transform the source into a violin. This is called timbre transfer ([try the demo notebook!](https://colab.research.google.com/github/magenta/ddsp/blob/master/ddsp/colab/demos/timbre_transfer.ipynb)).

`buildPlugins.m` constructs timbre transfer plugins for the four provided examples. The weights of a flute, violin, trumpet and saxophone model were extracted from the timbre transfer demo notebook. You can also build the plugins individually, if you call `addpath('plugincode')` first.

It's also possible to [train your own network](https://colab.research.google.com/github/magenta/ddsp/blob/master/ddsp/colab/demos/train_autoencoder.ipynb) with the same architecture as the ones in the timbre transfer demo, and use the `extract_weights.py` script to turn a checkpoint into a MAT file.

To use the weight files in a MATLAB plugin, simply inherit the `ddspPlugin` class and set the `ModelFile` property to the path of the MAT file containing the decoder weights. 

For further detail and a demonstration, refer to this video:

https://youtu.be/c_pZHzz_1bs

### Hybrid plugin
To generate a hybrid model, call `utils.combineWeights()` passing the filenames of two .mat files of decoder weights (and optionally two further numerical parameters representing the ratio between the weights). This will generate a file, `hybridWeights.mat`, that can be used with `generateAudioPlugin hybridPlugin`.

### Morphable plugin
A morphable plugin can be built with `generateAudioPlugin MorphablePlugin`. This plugin adds a slider with which the user can specify the ratio between decoder weights during processing. Specify the pair of weights files to be used when building the plugin by editing the constant properties of the `MorphableDecoder` class, e.g.

    weightfile1 = 'trumpetWeights.mat';
    weightfile2 = 'saxophoneWeights.mat';

### Issues
 - Audio drops out when looping an input audio sample, possibly due to MLPLayer and CircularBuffer responding poorly to periods of silence. Changing the Frame Size (or Morph) parameters while playback continues seems to wake things up again. Further investigation is required.

% plugin for a hybrid model
% Run utils.combineWeights() with the names of two weights .mat files to 
% generate a hybrid weights file to use with this plugin.

classdef hybridPlugin < ddspPlugin
    properties (Constant)
        ModelFile = 'hybridWeights.mat';
    end
end
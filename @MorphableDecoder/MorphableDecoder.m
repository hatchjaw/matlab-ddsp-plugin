classdef MorphableDecoder < handle
% MORPHABLEDECODER Construct the decoder part of the DDSP Autoencoder from files
% of weights.
%
%  This class loads its model weights from the supplied .mat files and 
%  constructs the inference-only decoder network of the DDSP Autoencoder.
%  
%  Given normalized values for loudness and pitch, the decoder predicts an 
%  amplitude, a harmonic distribution and a filter magnitude response that 
%  can be used by the SpectralModelingSynth to generate a frame of audio.
%
%  This decoder implements the RnnFcDecoder class from the DDSP python
%  code: Loudness and pitch values are passed through their respective
%  stacks of feed-forward layers (the number of layers is fixed here
%  because of coder restrictions). The output of the two stacks is then
%  concatenated and passed through a recurrent layer, whose output is used
%  to predict the synthesizer parameters.

    properties (Access = private)
        % Feed-forward stack for loudness
        LdLayer1, LdLayer2, LdLayer3;
        % Feed-forward stack for f0
        F0Layer1, F0Layer2, F0Layer3;
        % Recurrent layer  
        GRU;
        % Feed-forward stack for the GRU output
        OutLayer1, OutLayer2, OutLayer3;
        % Final projection layer
        OutProjKernel, OutProjBias;
    end

    properties (Constant)
        % It is necessary to declare the names of weights files as constants 
        % here, rather than in the MorphablePlugin class, otherwise, during
        % generateAudioPlugin, Matlab complains "This expression must be 
        % constant because its value determines the size or class of some 
        % expression".
        weightfile1 = 'trumpetWeights.mat';
        weightfile2 = 'saxophoneWeights.mat';
        % Number of harmonics for the additive synthesis.
        nHarmonics = 60;
        % Number of magnitudes for the filtered noise.
        nMagnitudes = 65;
    end
    
    methods
        function obj = MorphableDecoder()
            % Initialise the morphable decoder with empty layer instances.
            obj.LdLayer1 = MLPLayer();
            obj.LdLayer2 = MLPLayer();
            obj.LdLayer3 = MLPLayer();
            obj.F0Layer1 = MLPLayer();
            obj.F0Layer2 = MLPLayer();
            obj.F0Layer3 = MLPLayer();
            obj.OutLayer1 = MLPLayer();
            obj.OutLayer2 = MLPLayer();
            obj.OutLayer3 = MLPLayer();
            obj.GRU = GRULayer();
            
            % Trigger the initial layer update.
            obj.updateLayers();
        end
        
        function updateLayers(obj, morphRatio)
            % Update coefficients for layers of the neural network as a hybrid
            % of values from two weights files.

            if nargin == 1
                morphRatio = 50;
            end
            
            % Calculate the w1:w2 ratio.
            morphRatio = morphRatio / 100;
            magnitudeW1 = 1 - morphRatio;
            magnitudeW2 = morphRatio;

            % Create a placeholder struct for the hybrid weights.
            w = struct();
            
            % Load the weights files.
            w1 = coder.load(obj.weightfile1);
            w2 = coder.load(obj.weightfile2);
            
            % Iterate over the fields in the w1 and w2 structs, updating the 
            % hybrid struct with weights composed from the supplied weights,
            % combined with the morph ratio parameter.
            % NB, this assumes w2 has the same structure as w1, both in terms of
            % field names and vector dimensions.
            fields = fieldnames(w1);
            for k = 1:numel(fields)
                    w.(fields{k}) = magnitudeW1 * w1.(fields{k}) + ...
                        magnitudeW2 * w2.(fields{k});
            end
            
            obj.LdLayer1.update(w.ld_dense_0_kernel, w.ld_dense_0_bias,...
                w.ld_norm_0_beta, w.ld_norm_0_gamma);
            obj.LdLayer2.update(w.ld_dense_1_kernel, w.ld_dense_1_bias,...
                w.ld_norm_1_beta, w.ld_norm_1_gamma);
            obj.LdLayer3.update(w.ld_dense_2_kernel, w.ld_dense_2_bias,...
                w.ld_norm_2_beta, w.ld_norm_2_gamma);

            obj.F0Layer1.update(w.f0_dense_0_kernel, w.f0_dense_0_bias,...
                w.f0_norm_0_beta, w.f0_norm_0_gamma);
            obj.F0Layer2.update(w.f0_dense_1_kernel, w.f0_dense_1_bias,...
                w.f0_norm_1_beta, w.f0_norm_1_gamma);
            obj.F0Layer3.update(w.f0_dense_2_kernel, w.f0_dense_2_bias,...
                w.f0_norm_2_beta, w.f0_norm_2_gamma); 
            
            obj.OutLayer1.update(w.out_dense_0_kernel, w.out_dense_0_bias,...
                w.out_norm_0_beta, w.out_norm_0_gamma);
            obj.OutLayer2.update(w.out_dense_1_kernel, w.out_dense_1_bias,...
                w.out_norm_1_beta, w.out_norm_1_gamma);
            obj.OutLayer3.update(w.out_dense_2_kernel, w.out_dense_2_bias,...
                w.out_norm_2_beta, w.out_norm_2_gamma);
            
            obj.GRU.update(w.gru_kernel, w.gru_recurrent, w.gru_bias);
            
            obj.OutProjKernel = double(w.outsplit_kernel);
            obj.OutProjBias   = double(w.outsplit_bias);
        end
        
        function [amp, harmDist, noiseMags] = call(obj, ld, f0)
            % Predict synthesizer parameters for one frame of audio.
            % Inputs:
            %    ld   : normalized loudness in dB
            %    f0   : normalized pitch
            % 
            % Outputs:
            %    amp      : The overall amplitude for the additive synthesis
            %    harmDist : The amplitudes of the individual harmonics
            %    noiseMags: The magnitude response of the filter applied to
            %               white noise
            
            % pass inputs through feed-forward stacks
            ld = obj.LdLayer1.call(ld);
            ld = obj.LdLayer2.call(ld);
            ld = obj.LdLayer3.call(ld);

            f0 = obj.F0Layer1.call(f0);
            f0 = obj.F0Layer2.call(f0);
            f0 = obj.F0Layer3.call(f0);
            
            % concatenate and pass through recurrent layer
            out = [ld f0];
            out = [out obj.GRU.call(out)];
            
            % pass through output stack and projection
            out = obj.OutLayer1.call(out);
            out = obj.OutLayer2.call(out);
            out = obj.OutLayer3.call(out);
            out = out * obj.OutProjKernel + obj.OutProjBias;
            
            % extract parameters from projecteed output
            amp = out(1);
            harmDist = out(2:obj.nHarmonics+1);
            noiseMags = out(obj.nHarmonics+2:end);
        end
        
        function reset(obj)
            % Reset the internal state of the recurrent layer
            obj.GRU.reset;
        end
    end
end
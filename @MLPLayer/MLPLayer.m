classdef MLPLayer < handle
% MLPLAYER A single layer of an inference-only multi-layer perceptron. 
%
%  This layer form the building block of the feed-forward stack used in the 
%  decoder. It consists of a dense layer, followed by layer normalization,
%  followed by leaky ReLU.

    properties (Access = private)
        Kernel, Bias;         % weights of the dense layer
        Beta, Gamma, Epsilon; % weights for the layer normalization
        Alpha;                % factor of the leaky ReLU
    end
    
    methods
        function obj = MLPLayer(kernel, bias, beta, gamma, epsilon, alpha)
            % MLPLayer constructor
            % Called without parameters it just creates a placeholder instance
            % of the class. If provided with parameters (e.g. for fixed-model 
            % implementations) sets the properties of the instance -- see
            % MLPLayer.update().
            if nargin >= 4
                if nargin == 4
                    % The default values used in the tensorflow implementation
                    epsilon = 1e-3;
                    alpha = 0.2;
                end
                obj.update(kernel, bias, beta, gamma, epsilon, alpha);
            end
        end
        
        function update(obj, kernel, bias, beta, gamma, epsilon, alpha)
            % Initialize given weights
            
            % explicitly require double for code generation
            obj.Kernel = double(kernel);
            obj.Bias = double(bias);
            obj.Beta = double(beta);
            obj.Gamma = double(gamma);
            
            if nargin > 5
                obj.Epsilon = epsilon;
                obj.Alpha = alpha;
            else
                % The default values used in the tensorflow implementation
                obj.Epsilon = 1e-3;
                obj.Alpha = 0.2;
            end
        end
        
        function out = call(obj, in)
            % There appears to be a bug, whereby when loaded audio is looped the
            % second playthrough results in silent output... possibly due to
            % 'in' sometimes being a vector of ±Inf.
            
            % dense layer
            out = in * obj.Kernel + obj.Bias;
    
            % layer normalization
            
            % first normalize by mean and variance of current input
            out = (out - mean(out)) / sqrt(var(out) + obj.Epsilon);
            
            % then apply learned scaling and offset
            out = obj.Gamma .* out + obj.Beta;

            % Non-Linearity: Leaky ReLU
            out(out < 0) = out(out < 0) * obj.Alpha;
        end 
    end
end


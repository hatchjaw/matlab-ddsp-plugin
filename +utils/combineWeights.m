function combineWeights(weightsFile1, weightsFile2, varargin)
%COMBINEWEIGHTS Create a weights file from the data in two weights files.
%   Given two weights files (.mat), computes the average of each formatted
%   data entry and generates a new weights file containing the resulting
%   data.
w1 = load(weightsFile1);
w2 = load(weightsFile2);
w3 = {};

fields = fieldnames(w1);
doMagnitudes = nargin == 4 && isnumeric(varargin{1}) && isnumeric(varargin{2});
for k=1:numel(fields)
    if(doMagnitudes)
        w3.(fields{k}) = ( ...
            varargin{1} * w1.(fields{k}) + ...
            varargin{2} * w2.(fields{k}) ...
        ) / (varargin{1} + varargin{2});
    else
        w3.(fields{k}) = ( w1.(fields{k}) + w2.(fields{k}) ) / 2;
    end
end

save('hybridWeights.mat', '-struct', 'w3');
end


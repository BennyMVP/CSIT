function [varargout] = permuteToTFDimensionOrder(varargin)
% Copyright 2023-2025 The MathWorks, Inc.
% PERMUTETOTFDIMENSIONORDER This function permutes placeholder function inputs to forward TF Dimension order
varargout = cell(1, nargin-1);
inputRanks = varargin{end};
for i=1:nargin-1
    x = varargin{i};
    inputRank = inputRanks(i);
    % permute to forward TensorFlow ordering using input rank
    if inputRank > 1
        x = permute(x, inputRank:-1:1);
    end
    varargout{i} = x;
end
end
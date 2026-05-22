function [varargout] = permuteToReverseTFDimensionOrder(varargin)
import trainedSynthSegModel.ops.*;

% Copyright 2023-2025 The MathWorks, Inc.
% PERMUTETOREVERSETFDIMENSIONORDER This function permutes placeholder function outputs from
% forward TF dimension order to reverse TF Dimension order
varargout = cell(1, nargin-1);
outputRanks = varargin{end};
for i=1:nargin-1
    x = varargin{i};
    outputRank = outputRanks(i);
    % permute to reverse TensorFlow order using output rank
    if outputRank > 1
        x = iPermuteToReverseTF(x, outputRank, false);
    else
        x = dlarray(x);
    end
    varargout{i} = x;
end
end
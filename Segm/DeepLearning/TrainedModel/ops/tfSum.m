function y = tfSum(x, axis, keepDims, inputRanks)

% Copyright 2022-2024 The MathWorks, Inc.

xRank = inputRanks(1);

if any(axis < 0)
    % Handle negative axis values
    negIdx = axis < 0;
    axis(negIdx) = xRank + axis(negIdx);
end

% xval is in reverse TF format
MLAxis = xRank - axis;

% Reverse TensorFlow dimension order
if xRank <= 1
    y = sum(x(:), MLAxis);
else
    y = sum(x, MLAxis);
end

if nargin < 3
    keepDims = false;
end

if ~keepDims
    dimsToDrop = MLAxis;
    dimsToDrop(dimsToDrop > ndims(y)) = [];
    newSize = size(y);
    newSize(dimsToDrop)= [];

    if numel(newSize) == 1
        y = reshape(y, newSize, []);
    elseif numel(newSize) > 1
        y = reshape(y, newSize);
    end

end
end
function y = tfMax(x, axis, keepDims, inputRanks)

%   Copyright 2020-2024 The MathWorks, Inc.

xRank = inputRanks(1);

if any(axis < 0)
    % Handle negative axis values
    negIdx = axis < 0;
    axis(negIdx) = xRank + axis(negIdx);
end

% xval is in reverse TF format
MLAxis = xRank - axis;

% Apply max, this preserves all dimesions

if xRank <= 1
    y = max(x(:), [], MLAxis);
else
    y = max(x, [], MLAxis);
end

if ~keepDims
    outsize = ones(1, xRank);
    outsize(1:ndims(y)) = size(y);
    outsize(MLAxis) = [];
    if numel(outsize) < 1
        outsize = [1 1];
    end
    yRank = xRank - numel(MLAxis);

    % Reshape to the reduced dims
    if yRank > 1
        y = reshape(y, outsize);
    else
        y = reshape(y, [outsize 1]);
    end
end
end
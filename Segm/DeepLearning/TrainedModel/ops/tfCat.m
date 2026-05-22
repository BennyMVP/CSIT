function y = tfCat(axis, inputRanks, varargin)

%   Copyright 2020-2023 The MathWorks, Inc.

% All input tensors should have the same rank.
if numel(unique(inputRanks)) ~= 1
    error('tfCat: Ranks of all input tensors should match for ConcatV2.');
end

outputRank = inputRanks(1);

% If axis is a struct extract the numeric axis value
% Handle negative axis value
if axis < 0
    mlAxis = axis - floor(axis./outputRank).*outputRank;
else
    mlAxis = axis;
end

mlAxis = outputRank - mlAxis;

isDlarray = cellfun(@(x)isdlarray(x), varargin);
if any(isDlarray)
    % if any inputs are dlarrays, all values must be cast to the same
    % type and converted to TensorFlow dimension format.
    for i = 2:numel(varargin)
        varargin{i} = cast(varargin{i}, 'like', varargin{1});
    end
end

nonEmptyTensors = {};
j = 1;
for i = 1:numel(varargin)
    varargin{i} = varargin{i};
    % remove any empty tensors
    if ~isempty(varargin{i})
        nonEmptyTensors{j} = varargin{i}; %#ok<AGROW>
        j = j + 1;
    end
end

% concatenate all inputs (in reverse TF format)
% as long as one of the inputs has labels the output of 'cat' will have labels.
y = cat(mlAxis, nonEmptyTensors{:});
y = dlarray(y);
end
function dlVal = iAddDataFormatLabels(dlVal, dlRank)
% Adds data format labels to input dlarray, if it doesn't already have one.
% Takes as input a dlarray struct with value and rank information and outputs a labeled dlarray.

% Copyright 2022-2023 The MathWorks, Inc.

    if isdlarray(dlVal)
        if isempty(dlVal.dims) || all(dlVal.dims == 'U')
            [permvec, labels] = guessLabels(dlRank);
            dlVal = dlarray(permute(dlVal, permvec), labels);
        end
    else
    % For numeric data
        [permvec, labels] = guessLabels(dlVal,dlRank);
        dlVal = dlarray(permute(dlVal, permvec), labels);
    end
end

function [permvec, labels] = guessLabels(dlInRank)
    switch dlInRank
        case 4
            labels = "CSSB";
            permvec = [1 3 2 4];
        case 3
            labels = "CSS";
            permvec = [1 3 2];
        case 2
            labels = "CB";
            permvec = [1 2];
        otherwise
            error("Unable to determine data format labels for input tensor");
    end
end
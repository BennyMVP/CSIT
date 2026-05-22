function seg = postProcessBrainCANDIData(postIn,postFlipIn,imSize,cropIdx,metaData,classNames,labelIDs)
% Post process the data to get the segmentation maps. 2022).

% Apply 3D Gaussian smoothing
postIn = squeeze(extractdata(postIn));
sigma  = 0.5;
for n = 1:size(postIn,4)
    postIn(:,:,:,n) = imgaussfilt3(postIn(:,:,:,n),sigma, ...
        FilterSize=3,Padding=0);
end

if ~isempty(postFlipIn)
    postFlipIn = squeeze(extractdata(postFlipIn));
    for n = 1:size(postFlipIn,4)
        postFlipIn(:,:,:,n) = imgaussfilt3(postFlipIn(:,:,:,n), ...
            sigma,FilterSize=3,Padding=0);
    end
    postFlipIn = flip(postFlipIn,1);
    postFlipIn = flip(postFlipIn,2);
    postFlipIn = fliplr(postFlipIn);
    lrIndices = [2 3 4 5 6 7 8 9 10 11 15 16 17 18; 19 20 21 22 ...
                 23 24 25 26 27 28 29 30 31 32];
    rlIndices = flip(lrIndices);
    postFlipIn(:,:,:,reshape(lrIndices',1,[])) = ...
        postFlipIn(:,:,:,reshape(rlIndices',1,[]));
    postIn = 0.5*(postIn + postFlipIn);
end

% Use largest connected component
th   = 0.25;
temp = postIn(:,:,:,2:end);
postInMask = sum(temp,4) > th;
largeComp  = findLargestComponent(postInMask);
S = repmat(largeComp,1,1,1,size(temp,4));
temp(~S) = 0;
postIn(:,:,:,2:end) = temp;

% Make posteriors zero outside the largest connected component
% of each topological class
postInMask = postIn > th;
topology_classes = [0  4  4  5  5  6  6  7  8  9 10  1  2  3  5 11 ...
                    12 13 14 14 15 15 16 16 17 18 19 20 15 21 22 23];
for topology_class = unique(topology_classes(2:end))
    [~,ti] = find(topology_classes == topology_class);
    tmp    = postInMask(:,:,:,ti);
    tmp    = findLargestComponent(tmp);
    postIn(:,:,:,ti) = postIn(:,:,:,ti) .* tmp;
end

% Renormalize posteriors (robusto a divisione per zero / NaN)
s = sum(postIn,4);
s(s < 1e-9) = 1;
postIn = postIn ./ s;
postIn(isnan(postIn)) = 0;

% Hard segmentation
[~,segPatch] = max(postIn,[],4);

% Make segmentation maps to original image size
seg = ones(imSize);
seg(cropIdx(1)+1:cropIdx(4),cropIdx(2)+1:cropIdx(5), ...
    cropIdx(3)+1:cropIdx(6)) = segPatch;
seg = labelIDs(seg);

% Align prediction back to the input orientation
aff    = eye(4);
affRef = metaData.Transform.T';
seg    = alignBrainCANDIVolume(seg,aff,affRef);

% Convert segmentation result to categorical type
seg = categorical(seg,labelIDs,classNames);
end


function largeCompFinal = findLargestComponent(postInMask)
% Robusto a: maschere vuote, nessuna componente connessa trovata.
largeCompFinal = zeros(size(postInMask));
for ii = 1:size(postInMask,4)
    tmp = postInMask(:,:,:,ii);
    if ~any(tmp(:))            % maschera vuota: salta
        continue
    end
    CC = bwconncomp(tmp,6);
    if CC.NumObjects == 0      % nessuna componente: salta
        continue
    end
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [~,idx]   = max(numPixels);
    lc = false(size(tmp));
    lc(CC.PixelIdxList{idx}) = true;
    largeCompFinal(:,:,:,ii) = lc;
end
end
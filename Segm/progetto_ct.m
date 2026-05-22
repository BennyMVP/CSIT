% CT
clc; 
close all; 
clear all;

im = double(imread('segm2.png'));
figure, imagesc(im), axis image, colormap("gray"), colorbar
title('Immagine originale')

im = im - min(im(:));
im = im ./ max(im(:));

figure, imagesc(im), axis image, colormap("gray"), colorbar
title('Immagine normalizzata')
figure, histogram(im(:),100), grid
title('Istogramma immagine')

%% Segmentazione per soglie
th = [0 0.2 0.31 0.33 1];
sgm2 = zeros(size(im));
n_regioni = length(th)-1;
tic
for k = 1:n_regioni
    temp = double((im>=th(k)) & (im<th(k+1)));
    temp = bwareaopen(temp, 10);
    sgm_stack(:,:,k) = temp;
    sgm2 = sgm2 + (k-1)*temp;
end
t_threshold = toc;
figure, montage(255*sgm_stack), title('Segmentazione a soglie')
figure, imagesc(sgm2), axis image, colorbar, title('Mappa segmentata a soglie')

%% Segmentazione basata sui bordi
h = fspecial('sobel');
b_o = conv2(im, h, 'same');
b_v = conv2(im, h.', 'same');
b = abs(b_o) + abs(b_v);
figure, imagesc(b), axis image, colormap("gray"), colorbar
title('Segmentazione basata sui bordi')

%% Region growing CT
img = padarray(im,[1 1],-1000);
figure, imagesc(img,[0 1]); axis image, colormap(gray), colorbar, hold on
title('Seleziona un seed con il mouse')
[x, y] = ginput(1); x = round(x); y = round(y);
sgm3 = zeros(size(img)); sgm3(y,x) = 1;
h2 = image(cat(3, zeros(size(img)), zeros(size(img)), sgm3)); alpha(h2, 0.2)
th_rg = 0.0045;

tic
area_old = 0;                             
for k2 = 1:600
    temp2 = conv2(sgm3, ones(3), 'same');
    temp3 = (temp2 >= 1 & temp2 < 9);
    while sum(temp3(:)) > 0
        idx = find(temp3);
        for k = 1:length(idx)
            [r, c] = ind2sub(size(img), idx(k));
            mask_patch = sgm3(r+(-1:1), c+(-1:1)) > 0;
            val_patch  = img(r+(-1:1), c+(-1:1));
            vicini_regione = val_patch(mask_patch);
            if ~isempty(vicini_regione)
                if min(abs(img(r,c) - vicini_regione)) < th_rg
                    sgm3(r,c) = 1;
                end
            end
            temp3(idx(k)) = 0;
        end
        delete(h2); h2 = image(cat(3, zeros(size(img)), zeros(size(img)), sgm3)); alpha(h2, 0.2)
    end
    
    % Criterio di arresto                  
    sgm3_crop_temp = sgm3(2:end-1, 2:end-1);
    area_new = sum(sgm3_crop_temp(:));
    if area_new == area_old
        break;
    end
    area_old = area_new;
end
t_region_growing = toc;

sgm3_crop = sgm3(2:end-1, 2:end-1);
sgm3_crop = bwareaopen(sgm3_crop, 20);
sgm3_crop = imclose(sgm3_crop, strel('disk',2));
figure, imagesc(im), axis image, colormap(gray), colorbar, hold on
h3 = image(cat(3, 0.8*sgm3_crop, zeros(size(im)), 1.0*sgm3_crop)); alpha(h3, 0.35*sgm3_crop)
title('Risultato region growing')
%% Kmeans (K=3)

K = 4; 
centroidi = sort(0.45*rand(K,1));
N = 10;
tic
figure
for n = 1:N
    for k = 1:K, d(:,:,k) = abs(im - centroidi(k)); end
    [v, sgm4] = min(d, [], 3);
    imagesc(sgm4), axis image, title(n); drawnow
    for k = 1:K, centroidi(k) = mean(im(sgm4 == k)); end
end
t_kmeans = toc;

%% Analisi di K 
sgm4_K = zeros(size(im,1), size(im,2), 8);
V = zeros(1,8);
for Kk = 1:8
    val = im(:);
    centroidi = sort(val(randperm(length(val),Kk)));
    p = 2;
    
    if Kk == 4
        t_kmeans_conv = tic;   % <-- cronometro per K=4 con convergenza
    end
    
    while p > 1
        d = zeros(size(im,1), size(im,2), Kk);
        for k = 1:Kk
            d(:,:,k) = abs(im - centroidi(k));
        end
        [v, sgm4] = min(d, [], 3);
        centroidi_old = centroidi;
        for k = 1:Kk
            if any(sgm4(:) == k)
                centroidi(k) = mean(im(sgm4 == k));
            end
        end
        p = max(100 * abs(centroidi - centroidi_old) ./ max(abs(centroidi_old), eps));
    end
    
    if Kk == 4
        t_kmeans = toc(t_kmeans_conv);   % <-- salva tempo K=4
    end
    
    sgm4_K(:,:,Kk) = sgm4;
    V(Kk) = mean(v(:));
end
figure, stem(V), grid, title('Andamento della distanza media al variare di K')
figure
for Kk = 1:8
    subplot(2,4,Kk), imagesc(sgm4_K(:,:,Kk)), axis image, title(['K = ', num2str(Kk)])
end
figure, imagesc(sgm4_K(:,:,4)), axis image, title('4')

%% Ground truth
gt_file = 'gt_ct.mat';
if isfile(gt_file)
    load(gt_file, 'gt_mask');
    fprintf('\n>> Ground truth caricata da %s\n', gt_file);
else
    fprintf('\n>> Disegna a mano il contorno della struttura di riferimento.\n');
    figure('Name','Disegna la Ground Truth');
    imshow(im, []), title('Disegna il contorno, doppio click per chiudere')
    h_roi = drawpolygon('Color','r','LineWidth',2); wait(h_roi);
    gt_mask = createMask(h_roi);
    save(gt_file, 'gt_mask');
    fprintf('>> Ground truth salvata su %s\n', gt_file); close
end
figure('Name','Ground Truth'), imshow(im,[]), hold on
visboundaries(gt_mask,'Color','r','LineWidth',1.5); title('Ground Truth (contorno rosso)'), hold off

%% Bounding box
margin = 5;
stats = regionprops(gt_mask, 'BoundingBox'); bb = stats(1).BoundingBox;
x1 = max(1, floor(bb(1) - margin)); y1 = max(1, floor(bb(2) - margin));
x2 = min(size(im,2), ceil(bb(1) + bb(3) + margin)); y2 = min(size(im,1), ceil(bb(2) + bb(4) + margin));
roi = false(size(im)); roi(y1:y2, x1:x2) = true;
figure('Name','ROI di valutazione'); imshow(im,[]), hold on
visboundaries(gt_mask,'Color','r','LineWidth',1.5);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','y','LineWidth',2);
title(sprintf('GT (rosso) e bounding box (giallo), margine %d px',margin)), hold off

%% THRESHOLD
[mask_th, class_th, dice_th, prec_th, rec_th] = bestMatchClassROI_full(sgm2, gt_mask, roi);
iou_th = jaccard(gt_mask(roi), mask_th(roi));
hd_th  = hausdorffDist(gt_mask & roi, mask_th & roi);
hd95_th = hausdorffDist95(gt_mask & roi, mask_th & roi);

%% REGION GROWING
mask_rg = logical(sgm3_crop);
dice_rg = dice(gt_mask, mask_rg);
iou_rg  = jaccard(gt_mask, mask_rg);
hd_rg   = hausdorffDist(gt_mask, mask_rg);
hd95_rg = hausdorffDist95(gt_mask, mask_rg);
TP = sum(mask_rg(:) & gt_mask(:)); FP = sum(mask_rg(:) & ~gt_mask(:)); FN = sum(~mask_rg(:) & gt_mask(:));
prec_rg = TP / (TP + FP + eps); rec_rg = TP / (TP + FN + eps);

%% K-MEANS K=4
sgm_km = sgm4_K(:,:,4);
[mask_km, class_km, dice_km, prec_km, rec_km] = bestMatchClassROI_full(sgm_km, gt_mask, roi);
iou_km = jaccard(gt_mask(roi), mask_km(roi));
hd_km  = hausdorffDist(gt_mask & roi, mask_km & roi);
hd95_km = hausdorffDist95(gt_mask & roi, mask_km & roi);

%% Metriche
fprintf('\n================================================================================\n');
fprintf('  METRICHE - threshold/K-means valutati nella bounding box GT (margine %d px)\n', margin);
fprintf('================================================================================\n');
fprintf('%-22s | %-6s | %-6s | %-7s | %-7s | %-10s | %-8s\n', ...
    'Metodo','Dice','IoU','HD','HD95','Prec/Rec','Tempo(s)');
fprintf('%s\n', repmat('-',1,95));
fprintf('%-22s | %.4f | %.4f | %5.2f | %5.2f | %.3f/%.3f | %7.2f\n', ...
    sprintf('Threshold (cl.%d)*',class_th), dice_th, iou_th, hd_th, hd95_th, prec_th, rec_th, t_threshold);
fprintf('%-22s | %.4f | %.4f | %5.2f | %5.2f | %.3f/%.3f | %7.2f\n', ...
    'Region growing', dice_rg, iou_rg, hd_rg, hd95_rg, prec_rg, rec_rg, t_region_growing);
fprintf('%-22s | %.4f | %.4f | %5.2f | %5.2f | %.3f/%.3f | %7.2f\n', ...
    sprintf('K-means K=4 (cl.%d)*',class_km), dice_km, iou_km, hd_km, hd95_km, prec_km, rec_km, t_kmeans);
fprintf('%s\n', repmat('-',1,95));
fprintf('* metrica calcolata nella bounding box della GT (solo per threshold e K-means)\n');

%% Visualizzazione confronto (solo Dice sopra ogni sottogramma)
figure('Name','Confronto maschere','Position',[100 100 1300 350]);
subplot(1,4,1), imshow(im,[]), hold on
visboundaries(gt_mask,'Color','r','LineWidth',1.5);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','y','LineWidth',1,'LineStyle','--');
title('Ground Truth')

subplot(1,4,2), imshow(im,[]), hold on
visboundaries(mask_th & roi,'Color','y','LineWidth',1.5);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','y','LineWidth',1,'LineStyle','--');
title(sprintf('Threshold (Dice=%.2f)',dice_th))

subplot(1,4,3), imshow(im,[]), hold on
visboundaries(mask_rg,'Color','c','LineWidth',1.5);
title(sprintf('Region Growing (Dice=%.2f)',dice_rg))

subplot(1,4,4), imshow(im,[]), hold on
visboundaries(mask_km & roi,'Color','g','LineWidth',1.5);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','y','LineWidth',1,'LineStyle','--');
title(sprintf('K-means (Dice=%.2f)',dice_km))

%% FUNZIONI
function d = dice(A,B)
    if ~any(A(:)) && ~any(B(:)), d = 1; elseif ~any(A(:)) || ~any(B(:)), d = 0;
    else d = 2 * sum(A(:) & B(:)) / (sum(A(:)) + sum(B(:))); end
end

function [bestMask, bestClass, bestDice, bestPrec, bestRec] = bestMatchClassROI_full(labelMap, gt, roi)
    classes = unique(labelMap(:));
    bestDice = -1; bestClass = classes(1); bestMask = false(size(labelMap));
    bestPrec = 0; bestRec = 0;
    for c = classes(:).'
        m = (labelMap == c); m_roi = m & roi;
        if ~any(m_roi(:)), continue; end
        d = dice(gt(roi), m_roi(roi));
        if d > bestDice
            bestDice = d; bestClass = c; bestMask = m;
            TP = sum(gt(roi) & m_roi(roi)); FP = sum(~gt(roi) & m_roi(roi)); FN = sum(gt(roi) & ~m_roi(roi));
            bestPrec = TP / (TP + FP + eps); bestRec = TP / (TP + FN + eps);
        end
    end
end

function hd = hausdorffDist(A,B)
    if ~any(A(:)) || ~any(B(:)), hd = NaN; return; end
    d1 = bwdist(A); hd1 = max(d1(B)); d2 = bwdist(B); hd2 = max(d2(A));
    hd = max(hd1, hd2);
end

function hd95 = hausdorffDist95(A,B)
    if ~any(A(:)) || ~any(B(:)), hd95 = NaN; return; end
    d1 = bwdist(A); hd1 = prctile(d1(B), 95); d2 = bwdist(B); hd2 = prctile(d2(A), 95);
    hd95 = max(hd1, hd2);
end
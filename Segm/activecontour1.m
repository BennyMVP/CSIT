%% ACTIVE CONTOURS

clc; 
clear; 
close all;

% Parametri comuni
pausaAnimazione    = 0.005;  
metodo             = 'Chan-Vese';
semilatoROI        = 80;

%% CT
% Parametri per la CT
stepIter_CT        = 20;
nIterMax_CT        = 400;
raggioMaschera_CT  = 8;
smoothFactor_CT    = 1.0;
contractionBias_CT = 0.2;
areaTolleranza_CT  = 0.02;
finestraStabile_CT = 5;

fprintf('\n========== CT - sezione assiale del busto ==========\n');
I1 = imread('segm2.png');
if size(I1,3)==3, I1_gray = rgb2gray(I1); else, I1_gray = I1; end
I1_gray = im2double(I1_gray);

figure; imshow(I1_gray, []); title('CT: clicca all''interno della aorta')
[xClick1, yClick1] = ginput(1); xClick1 = round(xClick1); yClick1 = round(yClick1);

% ROI intorno al click
x1_min = max(1, xClick1 - semilatoROI); x1_max = min(size(I1_gray,2), xClick1 + semilatoROI);
y1_min = max(1, yClick1 - semilatoROI); y1_max = min(size(I1_gray,1), yClick1 + semilatoROI);
I1_roi = I1_gray(y1_min:y1_max, x1_min:x1_max);
xSeed1_roi = xClick1 - x1_min + 1; ySeed1_roi = yClick1 - y1_min + 1;
[righeROI1, colonneROI1] = size(I1_roi);
[X1, Y1] = meshgrid(1:colonneROI1, 1:righeROI1);
mask1_roi = (X1 - xSeed1_roi).^2 + (Y1 - ySeed1_roi).^2 <= raggioMaschera_CT^2;
mask1_roi = imfill(mask1_roi, 'holes');

% evoluzione CT
fig1 = figure('Name','Evoluzione CT'); imshow(I1_roi, []); hold on;
title('Evoluzione Active Contours - CT');

BW1_roi = mask1_roi;
areaPrec = sum(BW1_roi(:));
contStabile = 0;
iterEff = 0;

tic
for iter = stepIter_CT:stepIter_CT:nIterMax_CT
    BW1_roi = activecontour(I1_roi, BW1_roi, stepIter_CT, metodo, ...
        'SmoothFactor', smoothFactor_CT, 'ContractionBias', contractionBias_CT);
    iterEff = iter;

    areaNuova = sum(BW1_roi(:));
    variazioneArea = abs(areaNuova - areaPrec) / (areaPrec + eps);
    
    if mod(iter, 5*stepIter_CT) == 0
        fprintf('  CT iter %d: area = %.0f, var = %.4f\n', iter, areaNuova, variazioneArea);
    end

    if variazioneArea < areaTolleranza_CT
        contStabile = contStabile + 1;
        if contStabile >= finestraStabile_CT
            break;
        end
    else
        contStabile = 0;
    end
    areaPrec = areaNuova;

    cla; imshow(I1_roi, []); hold on;
    visboundaries(BW1_roi, 'Color', 'r', 'LineWidth', 1.5);
    plot(xSeed1_roi, ySeed1_roi, 'y+', 'MarkerSize', 12, 'LineWidth', 2);
    title(sprintf('Aorta | Iter %d (var=%.4f)', iter, variazioneArea));
    drawnow;
    if pausaAnimazione > 0, pause(pausaAnimazione); end
end
t1 = toc;
fprintf('  Arresto CT dopo %d iterazioni (area stabile)\n', iterEff);

% Pulizia e maschera finale
BW1_roi = bwareaopen(BW1_roi, 10);
BW1_roi = imclose(BW1_roi, strel('disk',1));
BW1_roi = imfill(BW1_roi, 'holes');
BW1_active = false(size(I1_gray));
BW1_active(y1_min:y1_max, x1_min:x1_max) = BW1_roi;

cc1 = bwconncomp(BW1_active);
label_map1 = labelmatrix(cc1);
seed_label1 = label_map1(yClick1, xClick1);
if seed_label1 > 0
    BW1_active = (label_map1 == seed_label1);
    fprintf('  Mantenuta componente connessa al seme (eliminate %d regioni isolate)\n', cc1.NumObjects-1);
else
    warning('Il seme non appartiene ad alcuna regione segmentata');
end

% Ground truth CT
gt_file_ct = 'gt_ct.mat';
if isfile(gt_file_ct)
    load(gt_file_ct, 'gt_mask');
else
    figure; imshow(I1_gray,[]); title('Disegna contorno della aorta');
    h = drawpolygon('Color','r','LineWidth',2); wait(h);
    gt_mask = createMask(h); save(gt_file_ct, 'gt_mask'); close;
end
gt_ct = gt_mask;

% Metriche CT
margin = 5;
stats = regionprops(gt_ct, 'BoundingBox'); bb = stats(1).BoundingBox;
x1_bb = max(1, floor(bb(1)-margin)); y1_bb = max(1, floor(bb(2)-margin));
x2_bb = min(size(I1_gray,2), ceil(bb(1)+bb(3)+margin));
y2_bb = min(size(I1_gray,1), ceil(bb(2)+bb(4)+margin));
roi_bb = false(size(I1_gray)); roi_bb(y1_bb:y2_bb, x1_bb:x2_bb) = true;
gt_roi = gt_ct & roi_bb; pred_roi = BW1_active & roi_bb;
dice1  = diceLocal(gt_roi, pred_roi);
iou1   = jaccard(gt_roi, pred_roi);
hd1    = hausdorffDist(gt_roi, pred_roi);
hd95_1 = hausdorffDist95(gt_roi, pred_roi);
TP = sum(gt_roi(:) & pred_roi(:));
FP = sum(~gt_roi(:) & pred_roi(:));
FN = sum(gt_roi(:) & ~pred_roi(:));
prec1 = TP/(TP+FP+eps); rec1 = TP/(TP+FN+eps);

%% MRI
% Parametri per MRI
stepIter_MRI        = 10;
nIterMax_MRI        = 600;
raggioMaschera_MRI  = 3;
smoothFactor_MRI    = 0;
contractionBias_MRI = -0.2;
areaTolleranza_MRI  = 0.01;
finestraStabile_MRI = 8;

fprintf('\n========== MRI - VENTRICOLO SINISTRO ==========\n');
I2 = imread('brain_mri1.png');
if size(I2,3)==3, I2_gray = rgb2gray(I2); else, I2_gray = I2; end
I2_gray = im2double(I2_gray);

figure; imshow(I2_gray, []);
title('MRI: clicca DENTRO il ventricolo sinistro (zona centrale)')
[xClick2, yClick2] = ginput(1); xClick2 = round(xClick2); yClick2 = round(yClick2);

x2_min = max(1, xClick2 - semilatoROI); x2_max = min(size(I2_gray,2), xClick2 + semilatoROI);
y2_min = max(1, yClick2 - semilatoROI); y2_max = min(size(I2_gray,1), yClick2 + semilatoROI);
I2_roi = I2_gray(y2_min:y2_max, x2_min:x2_max);
xSeed2_roi = xClick2 - x2_min + 1; ySeed2_roi = yClick2 - y2_min + 1;
[righeROI2, colonneROI2] = size(I2_roi);
[X2, Y2] = meshgrid(1:colonneROI2, 1:righeROI2);
mask2_roi = (X2 - xSeed2_roi).^2 + (Y2 - ySeed2_roi).^2 <= raggioMaschera_MRI^2;
mask2_roi = imfill(mask2_roi, 'holes');

fig2 = figure('Name','Evoluzione MRI'); imshow(I2_roi, []); hold on;
title('Evoluzione Active Contours - Ventricolo');

BW2_roi = mask2_roi;
areaPrec = sum(BW2_roi(:));
contStabile = 0; iterEff = 0;

tic
for iter = stepIter_MRI:stepIter_MRI:nIterMax_MRI
    BW2_roi = activecontour(I2_roi, BW2_roi, stepIter_MRI, metodo, ...
        'SmoothFactor', smoothFactor_MRI, 'ContractionBias', contractionBias_MRI);
    iterEff = iter;

    areaNuova = sum(BW2_roi(:));
    variazioneArea = abs(areaNuova - areaPrec) / (areaPrec + eps);
    
    if mod(iter, 5*stepIter_MRI) == 0
        fprintf('  MRI iter %d: area = %.0f, var = %.4f\n', iter, areaNuova, variazioneArea);
    end

    if variazioneArea < areaTolleranza_MRI
        contStabile = contStabile + 1;
        if contStabile >= finestraStabile_MRI
            break;
        end
    else
        contStabile = 0;
    end
    areaPrec = areaNuova;

    cla; imshow(I2_roi, []); hold on;
    visboundaries(BW2_roi, 'Color', 'r', 'LineWidth', 1.5);
    plot(xSeed2_roi, ySeed2_roi, 'y+', 'MarkerSize', 12, 'LineWidth', 2);
    title(sprintf('Ventricolo | Iter %d (var=%.4f)', iter, variazioneArea));
    drawnow;
    if pausaAnimazione > 0, pause(pausaAnimazione); end
end
t2 = toc;
fprintf('  Arresto MRI dopo %d iterazioni (area stabile)\n', iterEff);

BW2_roi = bwareaopen(BW2_roi, 10);
BW2_roi = imclose(BW2_roi, strel('disk',1));
BW2_roi = imfill(BW2_roi, 'holes');
BW2_active = false(size(I2_gray));
BW2_active(y2_min:y2_max, x2_min:x2_max) = BW2_roi;

cc2 = bwconncomp(BW2_active);
label_map2 = labelmatrix(cc2);
seed_label2 = label_map2(yClick2, xClick2);
if seed_label2 > 0
    BW2_active = (label_map2 == seed_label2);
    fprintf('  Mantenuta componente connessa al seme (eliminate %d regioni isolate)\n', cc2.NumObjects-1);
else
    warning('Il seme non appartiene ad alcuna regione segmentata');
end

gt_file_mri = 'gt_brain.mat';
if isfile(gt_file_mri)
    load(gt_file_mri, 'gt_mask');
else
    figure; imshow(I2_gray,[]); title('Disegna contorno ventricolo');
    h = drawpolygon('Color','r','LineWidth',2); wait(h);
    gt_mask = createMask(h); save(gt_file_mri, 'gt_mask'); close;
end
gt_mri = gt_mask;

stats = regionprops(gt_mri, 'BoundingBox'); bb = stats(1).BoundingBox;
x1_bb = max(1, floor(bb(1)-margin)); y1_bb = max(1, floor(bb(2)-margin));
x2_bb = min(size(I2_gray,2), ceil(bb(1)+bb(3)+margin));
y2_bb = min(size(I2_gray,1), ceil(bb(2)+bb(4)+margin));
roi_bb = false(size(I2_gray)); roi_bb(y1_bb:y2_bb, x1_bb:x2_bb) = true;
gt_roi = gt_mri & roi_bb; pred_roi = BW2_active & roi_bb;
dice2  = diceLocal(gt_roi, pred_roi);
iou2   = jaccard(gt_roi, pred_roi);
hd2    = hausdorffDist(gt_roi, pred_roi);
hd95_2 = hausdorffDist95(gt_roi, pred_roi);
TP = sum(gt_roi(:) & pred_roi(:));
FP = sum(~gt_roi(:) & pred_roi(:));
FN = sum(gt_roi(:) & ~pred_roi(:));
prec2 = TP/(TP+FP+eps); rec2 = TP/(TP+FN+eps);


fprintf('\n================================================================================\n');
fprintf('  METRICHE - ACTIVE CONTOURS (Chan-Vese) - bounding box GT (margine %d px)\n', margin);
fprintf('================================================================================\n');
fprintf('%-15s | %-6s | %-6s | %-7s | %-7s | %-12s | %-8s\n', ...
    'Immagine','Dice','IoU','HD','HD95','Prec/Rec','Tempo(s)');
fprintf('%s\n', repmat('-',1,95));
fprintf('%-15s | %.4f | %.4f | %5.2f | %5.2f | %.3f/%.3f | %7.2f\n', ...
    'Aorta (CT)',  dice1, iou1, hd1, hd95_1, prec1, rec1, t1);
fprintf('%-15s | %.4f | %.4f | %5.2f | %5.2f | %.3f/%.3f | %7.2f\n', ...
    'Ventricolo (MRI)', dice2, iou2, hd2, hd95_2, prec2, rec2, t2);
fprintf('%s\n', repmat('-',1,95));


figure('Name','Risultati Active Contours','Position',[100 100 1200 500]);
subplot(1,2,1); imshow(I1_gray,[]); hold on;
visboundaries(BW1_active, 'Color','r','LineWidth',1.5);
plot(xClick1, yClick1, 'y+', 'MarkerSize',10,'LineWidth',2);
title(sprintf('Aorta (Dice=%.3f, HD95=%.1f)', dice1, hd95_1));

subplot(1,2,2); imshow(I2_gray,[]); hold on;
visboundaries(BW2_active, 'Color','r','LineWidth',1.5);
plot(xClick2, yClick2, 'y+', 'MarkerSize',10,'LineWidth',2);
title(sprintf('Ventricolo (Dice=%.3f, HD95=%.1f)', dice2, hd95_2));

%funzioni
function d = diceLocal(A,B)
    if ~any(A(:)) && ~any(B(:)), d = 1;
    elseif ~any(A(:)) || ~any(B(:)), d = 0;
    else d = 2 * sum(A(:) & B(:)) / (sum(A(:)) + sum(B(:)));
    end
end

function hd = hausdorffDist(A,B)
    if ~any(A(:)) || ~any(B(:)), hd = NaN; return; end
    d1 = bwdist(A); hd1 = max(d1(B));
    d2 = bwdist(B); hd2 = max(d2(A));
    hd = max(hd1, hd2);
end

function hd95 = hausdorffDist95(A,B)
    if ~any(A(:)) || ~any(B(:)), hd95 = NaN; return; end
    d1 = bwdist(A); hd1 = prctile(d1(B), 95);
    d2 = bwdist(B); hd2 = prctile(d2(A), 95);
    hd95 = max(hd1, hd2);
end
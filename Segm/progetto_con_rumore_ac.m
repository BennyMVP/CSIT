%% ROBUSTEZZA AL RUMORE - ACTIVE CONTOURS
%si aggiunge rumore gaussiano bianco con sigma crescente
% all'immagine, si rilancia Chan-Vese con gli stessi parametri del codice
% senza rumore
%

clc;
clear all;
close all;

warning('off', 'all');  % disabilita tutti i warning


%% PARAMETRI PER IL RUMORE
sigma_vec = [0, 0.01, 0.05, 0.10];
rng(42);

% Parametri Chan-Vese
semilatoROI         = 80;
metodo              = 'Chan-Vese';

% CT
stepIter_CT         = 20;
nIterMax_CT         = 400;
raggioMaschera_CT   = 8;
smoothFactor_CT     = 1.0;
contractionBias_CT  = 0.2;
areaTolleranza_CT   = 0.02;
finestraStabile_CT  = 5;

% MRI
stepIter_MRI        = 10;
nIterMax_MRI        = 600;
raggioMaschera_MRI  = 3;
smoothFactor_MRI    = 0;
contractionBias_MRI = -0.2;
areaTolleranza_MRI  = 0.01;
finestraStabile_MRI = 8;

%%  CT
fprintf('\n========== CT - AORTA ==========\n');

I1 = imread('segm2.png');
if size(I1,3)==3, I1_gray = rgb2gray(I1); else, I1_gray = I1; end
I1_gray = im2double(I1_gray);

if ~isfile('gt_ct.mat'), error('gt_ct.mat non trovato'); end
load('gt_ct.mat', 'gt_mask'); gt_ct = gt_mask;

figure; imshow(I1_gray, []); title('CT: clicca DENTRO l''aorta (UNA volta)')
[xClick1, yClick1] = ginput(1);
xClick1 = round(xClick1); yClick1 = round(yClick1);
close;

% Bounding box GT per metriche
margin = 5;
stats = regionprops(gt_ct, 'BoundingBox'); bb = stats(1).BoundingBox;
x1_bb = max(1, floor(bb(1)-margin)); y1_bb = max(1, floor(bb(2)-margin));
x2_bb = min(size(I1_gray,2), ceil(bb(1)+bb(3)+margin));
y2_bb = min(size(I1_gray,1), ceil(bb(2)+bb(4)+margin));
roi_bb_ct = false(size(I1_gray)); roi_bb_ct(y1_bb:y2_bb, x1_bb:x2_bb) = true;

results_ct = struct('sigma',{},'dice',{},'hd95',{},'prec',{},'rec',{});
% Salviamo  i sigma
maschere_ct = cell(length(sigma_vec), 2);

for i = 1:length(sigma_vec)
    sigma = sigma_vec(i);
    fprintf('\n--- CT sigma=%.3f ---\n', sigma);

    rng(42 + i);
    I1_noisy = I1_gray + sigma * randn(size(I1_gray));
    I1_noisy = max(0, min(1, I1_noisy));

    % ROI attorno al click (stesso semilato)
    x1_min = max(1, xClick1 - semilatoROI); x1_max = min(size(I1_noisy,2), xClick1 + semilatoROI);
    y1_min = max(1, yClick1 - semilatoROI); y1_max = min(size(I1_noisy,1), yClick1 + semilatoROI);
    I1_roi = I1_noisy(y1_min:y1_max, x1_min:x1_max);
    xSeed_roi = xClick1 - x1_min + 1; ySeed_roi = yClick1 - y1_min + 1;

    % Maschera inizialE
    [X, Y] = meshgrid(1:size(I1_roi,2), 1:size(I1_roi,1));
    mask_roi = (X - xSeed_roi).^2 + (Y - ySeed_roi).^2 <= raggioMaschera_CT^2;

   
    BW_roi = mask_roi;
    areaPrec = sum(BW_roi(:)); contStabile = 0;
    for iter = stepIter_CT:stepIter_CT:nIterMax_CT
        BW_roi = activecontour(I1_roi, BW_roi, stepIter_CT, metodo, ...
            'SmoothFactor', smoothFactor_CT, 'ContractionBias', contractionBias_CT);
        areaNuova = sum(BW_roi(:));
        variaz = abs(areaNuova - areaPrec) / (areaPrec + eps);
        if variaz < areaTolleranza_CT
            contStabile = contStabile + 1;
            if contStabile >= finestraStabile_CT, break; end
        else
            contStabile = 0;
        end
        areaPrec = areaNuova;
    end

    % pulizia e ricomposizione
    BW_roi = bwareaopen(BW_roi, 10);
    BW_roi = imclose(BW_roi, strel('disk',1));
    BW_roi = imfill(BW_roi, 'holes');
    BW_active = false(size(I1_gray));
    BW_active(y1_min:y1_max, x1_min:x1_max) = BW_roi;

    % componente connessa al seme
    cc = bwconncomp(BW_active);
    label_map = labelmatrix(cc);
    seed_label = label_map(yClick1, xClick1);
    if seed_label > 0
        BW_active = (label_map == seed_label);
    else
        %se il seme non e' in nessuna componente: maschera vuota
        BW_active = false(size(I1_gray));
        fprintf('  [WARNING] Chan-Vese fallito: seme fuori dalla maschera segmentata\n');
    end

    % Metriche
    gt_roi = gt_ct & roi_bb_ct; pred_roi = BW_active & roi_bb_ct;
    d = diceLocal(gt_roi, pred_roi);
    h95 = hausdorffDist95(gt_roi, pred_roi);
    TP = sum(gt_roi(:) & pred_roi(:));
    FP = sum(~gt_roi(:) & pred_roi(:));
    FN = sum(gt_roi(:) & ~pred_roi(:));
    p = TP/(TP+FP+eps); r = TP/(TP+FN+eps);

    results_ct(i).sigma = sigma;
    results_ct(i).dice  = d;
    results_ct(i).hd95  = h95;
    results_ct(i).prec  = p;
    results_ct(i).rec   = r;

    fprintf('  Chan-Vese: Dice=%.3f  HD95=%.2f  Prec/Rec=%.3f/%.3f\n', d, h95, p, r);

    maschere_ct{i,1} = I1_noisy;
    maschere_ct{i,2} = BW_active;
end

%%  MRI
fprintf('\n========== MRI - VENTRICOLO ==========\n');

I2 = imread('brain_mri1.png');
if size(I2,3)==3, I2_gray = rgb2gray(I2); else, I2_gray = I2; end
I2_gray = im2double(I2_gray);

if ~isfile('gt_brain.mat'), error('gt_brain.mat non trovato'); end
load('gt_brain.mat', 'gt_mask'); gt_mri = gt_mask;

figure; imshow(I2_gray, []); title('MRI: clicca DENTRO il ventricolo (UNA volta)')
[xClick2, yClick2] = ginput(1);
xClick2 = round(xClick2); yClick2 = round(yClick2);
close;

stats = regionprops(gt_mri, 'BoundingBox'); bb = stats(1).BoundingBox;
x1_bb = max(1, floor(bb(1)-margin)); y1_bb = max(1, floor(bb(2)-margin));
x2_bb = min(size(I2_gray,2), ceil(bb(1)+bb(3)+margin));
y2_bb = min(size(I2_gray,1), ceil(bb(2)+bb(4)+margin));
roi_bb_mri = false(size(I2_gray)); roi_bb_mri(y1_bb:y2_bb, x1_bb:x2_bb) = true;

results_mri = struct('sigma',{},'dice',{},'hd95',{},'prec',{},'rec',{});
maschere_mri = cell(length(sigma_vec), 2);

for i = 1:length(sigma_vec)
    sigma = sigma_vec(i);
    fprintf('\n--- MRI sigma=%.3f ---\n', sigma);

    rng(42 + i);
    I2_noisy = I2_gray + sigma * randn(size(I2_gray));
    I2_noisy = max(0, min(1, I2_noisy));

    x2_min = max(1, xClick2 - semilatoROI); x2_max = min(size(I2_noisy,2), xClick2 + semilatoROI);
    y2_min = max(1, yClick2 - semilatoROI); y2_max = min(size(I2_noisy,1), yClick2 + semilatoROI);
    I2_roi = I2_noisy(y2_min:y2_max, x2_min:x2_max);
    xSeed_roi = xClick2 - x2_min + 1; ySeed_roi = yClick2 - y2_min + 1;

    [X, Y] = meshgrid(1:size(I2_roi,2), 1:size(I2_roi,1));
    mask_roi = (X - xSeed_roi).^2 + (Y - ySeed_roi).^2 <= raggioMaschera_MRI^2;

    BW_roi = mask_roi;
    areaPrec = sum(BW_roi(:)); contStabile = 0;
    for iter = stepIter_MRI:stepIter_MRI:nIterMax_MRI
        BW_roi = activecontour(I2_roi, BW_roi, stepIter_MRI, metodo, ...
            'SmoothFactor', smoothFactor_MRI, 'ContractionBias', contractionBias_MRI);
        areaNuova = sum(BW_roi(:));
        variaz = abs(areaNuova - areaPrec) / (areaPrec + eps);
        if variaz < areaTolleranza_MRI
            contStabile = contStabile + 1;
            if contStabile >= finestraStabile_MRI, break; end
        else
            contStabile = 0;
        end
        areaPrec = areaNuova;
    end

    BW_roi = bwareaopen(BW_roi, 10);
    BW_roi = imclose(BW_roi, strel('disk',1));
    BW_roi = imfill(BW_roi, 'holes');
    BW_active = false(size(I2_gray));
    BW_active(y2_min:y2_max, x2_min:x2_max) = BW_roi;

    cc = bwconncomp(BW_active);
    label_map = labelmatrix(cc);
    seed_label = label_map(yClick2, xClick2);
    if seed_label > 0
        BW_active = (label_map == seed_label);
    else
        BW_active = false(size(I2_gray));
        fprintf('  [WARNING] Chan-Vese fallito: seme fuori dalla maschera segmentata\n');
    end

    gt_roi = gt_mri & roi_bb_mri; pred_roi = BW_active & roi_bb_mri;
    d = diceLocal(gt_roi, pred_roi);
    h95 = hausdorffDist95(gt_roi, pred_roi);
    TP = sum(gt_roi(:) & pred_roi(:));
    FP = sum(~gt_roi(:) & pred_roi(:));
    FN = sum(gt_roi(:) & ~pred_roi(:));
    p = TP/(TP+FP+eps); r = TP/(TP+FN+eps);

    results_mri(i).sigma = sigma;
    results_mri(i).dice  = d;
    results_mri(i).hd95  = h95;
    results_mri(i).prec  = p;
    results_mri(i).rec   = r;

    fprintf('  Chan-Vese: Dice=%.3f  HD95=%.2f  Prec/Rec=%.3f/%.3f\n', d, h95, p, r);

    maschere_mri{i,1} = I2_noisy;
    maschere_mri{i,2} = BW_active;
end

%%  TABELLA
fprintf('\n\n================================================================================\n');
fprintf('  ROBUSTEZZA AL RUMORE - ACTIVE CONTOURS (Chan-Vese)\n');
fprintf('================================================================================\n');
fprintf('%-7s | %-6s | %-6s | %-12s || %-6s | %-6s | %-12s\n', ...
    'sigma','Dice','HD95','Prec/Rec (CT)','Dice','HD95','Prec/Rec (MRI)');
fprintf('%s\n', repmat('-',1,90));
for i = 1:length(sigma_vec)
    fprintf('%.3f   | %.3f  | %5.2f  | %.3f/%.3f  || %.3f  | %5.2f  | %.3f/%.3f\n', ...
        sigma_vec(i), ...
        results_ct(i).dice, results_ct(i).hd95, results_ct(i).prec, results_ct(i).rec, ...
        results_mri(i).dice, results_mri(i).hd95, results_mri(i).prec, results_mri(i).rec);
end
fprintf('%s\n', repmat('-',1,90));

%%  GRAFICO 
dice_ct_vec  = arrayfun(@(s) s.dice, results_ct);
dice_mri_vec = arrayfun(@(s) s.dice, results_mri);

figure('Name','Active Contours - robustezza al rumore','Position',[100 100 1100 500]);
plot(sigma_vec, dice_mri_vec, '-s', 'LineWidth', 2.5, 'MarkerSize', 12, ...
    'MarkerFaceColor', 'b', 'Color', 'b', 'DisplayName', 'Ventricolo (MRI)'); hold on
plot(sigma_vec, dice_ct_vec, '-o', 'LineWidth', 2.5, 'MarkerSize', 12, ...
    'MarkerFaceColor', [0.85 0.33 0.10], 'Color', [0.85 0.33 0.10], 'DisplayName', 'Aorta (CT)');

% Annotazioni numeriche su ogni punto
for i = 1:length(sigma_vec)
    text(sigma_vec(i), dice_mri_vec(i)+0.05, sprintf('%.3f', dice_mri_vec(i)), ...
        'HorizontalAlignment','center', 'FontSize', 10, 'Color', 'b', 'FontWeight','bold');
    text(sigma_vec(i), dice_ct_vec(i)-0.05, sprintf('%.3f', dice_ct_vec(i)), ...
        'HorizontalAlignment','center', 'FontSize', 10, 'Color', [0.85 0.33 0.10], 'FontWeight','bold');
end

xlabel('\sigma (rumore gaussiano)', 'FontSize', 13);
ylabel('Dice', 'FontSize', 13);
title('Chan-Vese: degradazione del Dice vs rumore', 'FontSize', 14);
grid on;
ylim([-0.05 1.10]);
xlim([-0.005 max(sigma_vec)+0.01]);
xticks(sigma_vec);  
legend('Location','best','FontSize',12);

%%  risultati
figure('Name','Chan-Vese: degradazione visiva (tutti i sigma)', ...
    'Position',[50 50 1600 700]);

for j = 1:length(sigma_vec)
    sigma = sigma_vec(j);

    % --- RIGA 1: VENTRICOLO MRI ---
    subplot(2, length(sigma_vec), j);
    imshow(maschere_mri{j,1}, []); hold on;
    visboundaries(gt_mri, 'Color','r','LineWidth',1.5);

    if any(maschere_mri{j,2}(:))
        visboundaries(maschere_mri{j,2}, 'Color','c','LineWidth',1.5);
        title(sprintf('Ventricolo \\sigma=%.2f  Dice=%.3f', sigma, results_mri(j).dice), ...
            'FontSize', 11);
    else
        % Maschera vuota: niente da disegnare in ciano
        % Mostra il punto del seed con marker giallo
        plot(xClick2, yClick2, 'y+', 'MarkerSize', 14, 'LineWidth', 2);
        title(sprintf('Ventricolo \\sigma=%.2f  FALLIMENTO (Dice=0)', sigma), ...
            'FontSize', 11, 'Color', 'r');
    end

    % --- RIGA 2: AORTA CT ---
    subplot(2, length(sigma_vec), j + length(sigma_vec));
    imshow(maschere_ct{j,1}, []); hold on;
    visboundaries(gt_ct, 'Color','r','LineWidth',1.5);

    if any(maschere_ct{j,2}(:))
        visboundaries(maschere_ct{j,2}, 'Color','c','LineWidth',1.5);
        title(sprintf('Aorta \\sigma=%.2f  Dice=%.3f', sigma, results_ct(j).dice), ...
            'FontSize', 11);
    else
        plot(xClick1, yClick1, 'y+', 'MarkerSize', 14, 'LineWidth', 2);
        title(sprintf('Aorta \\sigma=%.2f  FALLIMENTO (Dice=0)', sigma), ...
            'FontSize', 11, 'Color', 'r');
    end
end
sgtitle('Chan-Vese: degradazione visiva al crescere del rumore (GT rosso, predizione ciano, seed giallo se fallimento)', ...
    'FontSize', 12);


%%  FUNZIONI
function d = diceLocal(A, B)
    if ~any(A(:)) && ~any(B(:)), d = 1;
    elseif ~any(A(:)) || ~any(B(:)), d = 0;
    else, d = 2 * sum(A(:) & B(:)) / (sum(A(:)) + sum(B(:)));
    end
end

function hd95 = hausdorffDist95(A, B)
    if ~any(A(:)) || ~any(B(:)), hd95 = NaN; return; end
    d1 = bwdist(A); hd1 = prctile(d1(B), 95);
    d2 = bwdist(B); hd2 = prctile(d2(A), 95);
    hd95 = max(hd1, hd2);
end
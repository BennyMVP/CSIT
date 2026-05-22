%% ROBUSTEZZA AL RUMORE - METODI CLASSICI (Brain MRI + CT)
%
%  si aggiunge rumore gaussiano bianco con sigma crescente
% all'immagine preprocessata, si rilanciano i tre metodi classici
% (threshold, region growing, K-means) senza modificare nessuna soglia,
% si misurano Dice/IoU/HD95 e si plotta la degradazione.

clc; 
clear all; 
close all;

%% PARAMETRI PER IL RUMORE
sigma_vec = [0, 0.01, 0.05, 0.10];
rng(42);  % per riproducibilita' del rumore e del K-means

%% ========================== BRAIN MRI ==========================
fprintf('\n========== BRAIN MRI ==========\n');

% Carica immagine e preprocessing (gamma 0.5)
im_brain = double(imread('brain_mri1.png'));
im_brain = im_brain - min(im_brain(:));
im_brain = im_brain ./ max(im_brain(:));
im_brain = im_brain.^0.5;

% Carica GT
if ~isfile('gt_brain.mat')
    error('gt_brain.mat non trovato. Esegui prima progetto_brain_mri.m per creare la GT.');
end
load('gt_brain.mat', 'gt_mask');
gt_brain = gt_mask;

% Bounding box per le metriche localizzate
margin = 5;
[roi_brain, x1b, y1b, x2b, y2b] = makeBBoxROI(gt_brain, margin);

figure; imshow(im_brain, []); title('BRAIN: clicca seed per region growing (DENTRO il ventricolo)');
[xs_brain, ys_brain] = ginput(1);
xs_brain = round(xs_brain); ys_brain = round(ys_brain);
close;

% Parametri brain
th_thresh_brain = [0 0.45 0.7 1];
th_rg_brain = 0.015;
K_brain = 3;

% Loop sui livelli di rumore
nSigma = length(sigma_vec);
results_brain = zeros(nSigma, 3, 3);  % [sigma, metodo, metrica (dice, hd95, prec, rec)]
results_brain = struct('sigma',{},'dice',{},'hd95',{},'prec',{},'rec',{});

maschere_brain = cell(nSigma, 4);  % {sigma, gt+th+rg+km} per visualizzazione finale

for i = 1:nSigma
    sigma = sigma_vec(i);
    fprintf('\n--- BRAIN sigma=%.3f ---\n', sigma);

    % Aggiungi rumore gaussiano (poi clip a [0,1] per realismo)
    rng(42 + i);  % seed riproducibile per il rumore
    im_noisy = im_brain + sigma * randn(size(im_brain));
    im_noisy = max(0, min(1, im_noisy));

    % --- 1. THRESHOLD ---
    sgm_th = applyThreshold(im_noisy, th_thresh_brain);
    [mask_th, ~, dice_th, prec_th, rec_th] = bestMatchClassROI(sgm_th, gt_brain, roi_brain);
    hd95_th = hausdorffDist95(gt_brain & roi_brain, mask_th & roi_brain);

    % --- 2. REGION GROWING (stesso seed, senza visualizzazione) ---
    mask_rg = regionGrowingSilent(im_noisy, xs_brain, ys_brain, th_rg_brain);
    dice_rg = diceLocal(gt_brain, mask_rg);
    hd95_rg = hausdorffDist95(gt_brain, mask_rg);
    TP = sum(mask_rg(:) & gt_brain(:));
    FP = sum(mask_rg(:) & ~gt_brain(:));
    FN = sum(~mask_rg(:) & gt_brain(:));
    prec_rg = TP/(TP+FP+eps); rec_rg = TP/(TP+FN+eps);

    % --- 3. K-MEANS K=3 ---
    rng(42);  % seed identico per ogni sigma -> variabilita' solo dal rumore
    sgm_km = kmeansSilent(im_noisy, K_brain, 15);
    [mask_km, ~, dice_km, prec_km, rec_km] = bestMatchClassROI(sgm_km, gt_brain, roi_brain);
    hd95_km = hausdorffDist95(gt_brain & roi_brain, mask_km & roi_brain);

    % Salva risultati
    results_brain(i).sigma = sigma;
    results_brain(i).dice  = [dice_th, dice_rg, dice_km];
    results_brain(i).hd95  = [hd95_th, hd95_rg, hd95_km];
    results_brain(i).prec  = [prec_th, prec_rg, prec_km];
    results_brain(i).rec   = [rec_th,  rec_rg,  rec_km];

    fprintf('  Threshold:      Dice=%.3f  HD95=%.2f  Prec/Rec=%.3f/%.3f\n', dice_th, hd95_th, prec_th, rec_th);
    fprintf('  Region Growing: Dice=%.3f  HD95=%.2f  Prec/Rec=%.3f/%.3f\n', dice_rg, hd95_rg, prec_rg, rec_rg);
    fprintf('  K-means K=3:    Dice=%.3f  HD95=%.2f  Prec/Rec=%.3f/%.3f\n', dice_km, hd95_km, prec_km, rec_km);

    % Salva per visualizzazione (solo sigma=0 e sigma=0.10)
    if sigma == 0 || sigma == 0.10
        maschere_brain{i,1} = im_noisy;
        maschere_brain{i,2} = mask_th & roi_brain;
        maschere_brain{i,3} = mask_rg;
        maschere_brain{i,4} = mask_km & roi_brain;
    end
end

%% CT
fprintf('\n========== CT ==========\n');

im_ct = double(imread('segm2.png'));
im_ct = im_ct - min(im_ct(:));
im_ct = im_ct ./ max(im_ct(:));

if ~isfile('gt_ct.mat')
    error('gt_ct.mat non trovato. Esegui prima progetto_ct.m per creare la GT.');
end
load('gt_ct.mat', 'gt_mask');
gt_ct = gt_mask;

[roi_ct, x1c, y1c, x2c, y2c] = makeBBoxROI(gt_ct, margin);

figure; imshow(im_ct, []); title('CT: clicca seed per region growing (DENTRO l''aorta)');
[xs_ct, ys_ct] = ginput(1);
xs_ct = round(xs_ct); ys_ct = round(ys_ct);
close;

% Parametri CT
th_thresh_ct = [0 0.2 0.31 0.33 1];
th_rg_ct = 0.0045;
K_ct = 3;

results_ct = struct('sigma',{},'dice',{},'hd95',{},'prec',{},'rec',{});
maschere_ct = cell(nSigma, 4);

for i = 1:nSigma
    sigma = sigma_vec(i);
    fprintf('\n--- CT sigma=%.3f ---\n', sigma);

    rng(42 + i);
    im_noisy = im_ct + sigma * randn(size(im_ct));
    im_noisy = max(0, min(1, im_noisy));

    % Threshold
    sgm_th = applyThreshold(im_noisy, th_thresh_ct);
    [mask_th, ~, dice_th, prec_th, rec_th] = bestMatchClassROI(sgm_th, gt_ct, roi_ct);
    hd95_th = hausdorffDist95(gt_ct & roi_ct, mask_th & roi_ct);

    % Region growing
    mask_rg = regionGrowingSilent(im_noisy, xs_ct, ys_ct, th_rg_ct);
    dice_rg = diceLocal(gt_ct, mask_rg);
    hd95_rg = hausdorffDist95(gt_ct, mask_rg);
    TP = sum(mask_rg(:) & gt_ct(:));
    FP = sum(mask_rg(:) & ~gt_ct(:));
    FN = sum(~mask_rg(:) & gt_ct(:));
    prec_rg = TP/(TP+FP+eps); rec_rg = TP/(TP+FN+eps);

    % K-means
    rng(42);
    sgm_km = kmeansSilent(im_noisy, K_ct, 10);
    [mask_km, ~, dice_km, prec_km, rec_km] = bestMatchClassROI(sgm_km, gt_ct, roi_ct);
    hd95_km = hausdorffDist95(gt_ct & roi_ct, mask_km & roi_ct);

    results_ct(i).sigma = sigma;
    results_ct(i).dice  = [dice_th, dice_rg, dice_km];
    results_ct(i).hd95  = [hd95_th, hd95_rg, hd95_km];
    results_ct(i).prec  = [prec_th, prec_rg, prec_km];
    results_ct(i).rec   = [rec_th,  rec_rg,  rec_km];

    fprintf('  Threshold:      Dice=%.3f  HD95=%.2f  Prec/Rec=%.3f/%.3f\n', dice_th, hd95_th, prec_th, rec_th);
    fprintf('  Region Growing: Dice=%.3f  HD95=%.2f  Prec/Rec=%.3f/%.3f\n', dice_rg, hd95_rg, prec_rg, rec_rg);
    fprintf('  K-means K=3:    Dice=%.3f  HD95=%.2f  Prec/Rec=%.3f/%.3f\n', dice_km, hd95_km, prec_km, rec_km);

    if sigma == 0 || sigma == 0.10
        maschere_ct{i,1} = im_noisy;
        maschere_ct{i,2} = mask_th & roi_ct;
        maschere_ct{i,3} = mask_rg;
        maschere_ct{i,4} = mask_km & roi_ct;
    end
end

%%  TABELLE
fprintf('\n\n================================================================================\n');
fprintf('  ROBUSTEZZA AL RUMORE - DICE per metodo e sigma\n');
fprintf('================================================================================\n');
fprintf('%-7s | %-8s | %-8s | %-8s || %-8s | %-8s | %-8s\n', ...
    'sigma','TH (br)','RG (br)','KM (br)','TH (ct)','RG (ct)','KM (ct)');
fprintf('%s\n', repmat('-',1,80));
for i = 1:nSigma
    fprintf('%.3f   | %.4f   | %.4f   | %.4f   || %.4f   | %.4f   | %.4f\n', ...
        sigma_vec(i), ...
        results_brain(i).dice(1), results_brain(i).dice(2), results_brain(i).dice(3), ...
        results_ct(i).dice(1),    results_ct(i).dice(2),    results_ct(i).dice(3));
end
fprintf('%s\n', repmat('-',1,80));

%%  GRAFICI 
dice_brain = zeros(nSigma, 3);
dice_ct    = zeros(nSigma, 3);
for i = 1:nSigma
    dice_brain(i,:) = results_brain(i).dice;
    dice_ct(i,:)    = results_ct(i).dice;
end

figure('Name','Robustezza al rumore - Dice vs sigma','Position',[100 100 1200 450]);

subplot(1,2,1);
plot(sigma_vec, dice_brain(:,1), '-o', 'LineWidth', 2, 'DisplayName', 'Threshold'); hold on
plot(sigma_vec, dice_brain(:,2), '-s', 'LineWidth', 2, 'DisplayName', 'Region Growing');
plot(sigma_vec, dice_brain(:,3), '-^', 'LineWidth', 2, 'DisplayName', 'K-means K=3');
xlabel('\sigma (rumore gaussiano)'); ylabel('Dice');
title('BRAIN MRI: degradazione vs rumore');
grid on; ylim([0 1]); legend('Location','southwest','FontSize',10);

subplot(1,2,2);
plot(sigma_vec, dice_ct(:,1), '-o', 'LineWidth', 2, 'DisplayName', 'Threshold'); hold on
plot(sigma_vec, dice_ct(:,2), '-s', 'LineWidth', 2, 'DisplayName', 'Region Growing');
plot(sigma_vec, dice_ct(:,3), '-^', 'LineWidth', 2, 'DisplayName', 'K-means K=3');
xlabel('\sigma (rumore gaussiano)'); ylabel('Dice');
title('CT: degradazione vs rumore');
grid on; ylim([0 1]); legend('Location','southwest','FontSize',10);

%%  CONFRONTO
idx0 = find(sigma_vec == 0);
idx_high = find(sigma_vec == 0.10);

figure('Name','BRAIN - confronto sigma=0 vs sigma=0.10','Position',[100 100 1400 600]);
labels = {'GT (overlay)','Threshold','Region Growing','K-means'};
for j = 1:4
    subplot(2,4,j); imshow(maschere_brain{idx0,1}, []); hold on;
    if j == 1
        visboundaries(gt_brain, 'Color','r','LineWidth',1.5);
    else
        visboundaries(maschere_brain{idx0,j}, 'Color',colorMethod(j),'LineWidth',1.5);
    end
    title(sprintf('%s, \\sigma=0 (Dice=%.2f)', labels{j}, getDiceAt(results_brain, idx0, j)));
end
for j = 1:4
    subplot(2,4,j+4); imshow(maschere_brain{idx_high,1}, []); hold on;
    if j == 1
        visboundaries(gt_brain, 'Color','r','LineWidth',1.5);
    else
        visboundaries(maschere_brain{idx_high,j}, 'Color',colorMethod(j),'LineWidth',1.5);
    end
    title(sprintf('%s, \\sigma=0.10 (Dice=%.2f)', labels{j}, getDiceAt(results_brain, idx_high, j)));
end
sgtitle('BRAIN MRI: degradazione visiva con rumore');

% CT
figure('Name','CT - confronto sigma=0 vs sigma=0.10','Position',[100 100 1400 600]);
for j = 1:4
    subplot(2,4,j); imshow(maschere_ct{idx0,1}, []); hold on;
    if j == 1
        visboundaries(gt_ct, 'Color','r','LineWidth',1.5);
    else
        visboundaries(maschere_ct{idx0,j}, 'Color',colorMethod(j),'LineWidth',1.5);
    end
    title(sprintf('%s, \\sigma=0 (Dice=%.2f)', labels{j}, getDiceAt(results_ct, idx0, j)));
end
for j = 1:4
    subplot(2,4,j+4); imshow(maschere_ct{idx_high,1}, []); hold on;
    if j == 1
        visboundaries(gt_ct, 'Color','r','LineWidth',1.5);
    else
        visboundaries(maschere_ct{idx_high,j}, 'Color',colorMethod(j),'LineWidth',1.5);
    end
    title(sprintf('%s, \\sigma=0.10 (Dice=%.2f)', labels{j}, getDiceAt(results_ct, idx_high, j)));
end
sgtitle('CT: degradazione visiva con rumore');


%%  FUNZIONI

function [roi, x1, y1, x2, y2] = makeBBoxROI(gt_mask, margin)
    stats = regionprops(gt_mask, 'BoundingBox');
    bb = stats(1).BoundingBox;
    x1 = max(1, floor(bb(1) - margin));
    y1 = max(1, floor(bb(2) - margin));
    x2 = min(size(gt_mask,2), ceil(bb(1) + bb(3) + margin));
    y2 = min(size(gt_mask,1), ceil(bb(2) + bb(4) + margin));
    roi = false(size(gt_mask));
    roi(y1:y2, x1:x2) = true;
end

function sgm = applyThreshold(im, th)
    sgm = zeros(size(im));
    for k = 1:length(th)-1
        temp = double((im >= th(k)) & (im < th(k+1)));
        temp = bwareaopen(temp, 10);
        sgm = sgm + (k-1) * temp;
    end
end

function mask = regionGrowingSilent(im, xs, ys, th_rg)
% Region growing senza visualizzazione, stesso seed e soglia
    img = padarray(im, [1 1], -1000);
    sgm3 = zeros(size(img));
    sgm3(ys+1, xs+1) = 1;   % +1 per il padding
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
                vicini = val_patch(mask_patch);
                if ~isempty(vicini)
                    if min(abs(img(r,c) - vicini)) < th_rg
                        sgm3(r,c) = 1;
                    end
                end
                temp3(idx(k)) = 0;
            end
        end
        area_new = sum(sum(sgm3(2:end-1, 2:end-1)));
        if area_new == area_old, break; end
        area_old = area_new;
    end
    mask = sgm3(2:end-1, 2:end-1);
    mask = bwareaopen(mask, 20);
    mask = imclose(mask, strel('disk',2));
    mask = logical(mask);
end

function sgm = kmeansSilent(im, K, N)
% K-means senza visualizzazione, N iterazioni
    val = im(:);
    centroidi = sort(val(randperm(length(val),K)));
    for n = 1:N
        d = zeros(size(im,1), size(im,2), K);
        for k = 1:K
            d(:,:,k) = abs(im - centroidi(k));
        end
        [~, sgm] = min(d, [], 3);
        for k = 1:K
            if any(sgm(:) == k)
                centroidi(k) = mean(im(sgm == k));
            end
        end
    end
end

function [bestMask, bestClass, bestDice, bestPrec, bestRec] = bestMatchClassROI(labelMap, gt, roi)
    classes = unique(labelMap(:));
    bestDice = -1; bestClass = classes(1); bestMask = false(size(labelMap));
    bestPrec = 0; bestRec = 0;
    for c = classes(:).'
        m = (labelMap == c);
        m_roi = m & roi;
        if ~any(m_roi(:)), continue; end
        d = diceLocal(gt(roi), m_roi(roi));
        if d > bestDice
            bestDice = d; bestClass = c; bestMask = m;
            TP = sum(m_roi(roi) & gt(roi));
            FP = sum(m_roi(roi) & ~gt(roi));
            FN = sum(~m_roi(roi) & gt(roi));
            bestPrec = TP / (TP + FP + eps);
            bestRec  = TP / (TP + FN + eps);
        end
    end
end

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

function c = colorMethod(j)
% colori:  (threshold=yellow, RG=cyan, KM=green)
    switch j
        case 2, c = 'y';
        case 3, c = 'c';
        case 4, c = 'g';
        otherwise, c = 'r';
    end
end

function d = getDiceAt(results, i, j)
    if j == 1
        d = NaN;
    else
        d = results(i).dice(j-1);
    end
end
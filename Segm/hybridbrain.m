%% SEGMENTAZIONE IBRIDA MRI

clc;
clear;
close all;

%% Lettura e normalizzazione

im = imread('brain_mri1.png');

if size(im,3) == 3
    im = rgb2gray(im);
end

im = double(im);

im = im - min(im(:));
im = im ./ max(im(:));

%% Pre-processing

im_eq = im.^0.5;

%% Scelta del punto iniziale

figure;
imagesc(im_eq);
axis image;
colormap gray;
colorbar;
title('Clicca un punto dentro la regione da segmentare');

[x0, y0] = ginput(1);

x0 = round(x0);
y0 = round(y0);

x0 = max(1, min(x0, size(im_eq,2)));
y0 = max(1, min(y0, size(im_eq,1)));

fprintf('\nSeed selezionato: x = %d, y = %d, intensita = %.4f\n', ...
    x0, y0, im_eq(y0,x0));

%% Creazione ROI

raggio_roi = 80;

r1 = max(1, y0 - raggio_roi);
r2 = min(size(im_eq,1), y0 + raggio_roi);
c1 = max(1, x0 - raggio_roi);
c2 = min(size(im_eq,2), x0 + raggio_roi);

roi = im_eq(r1:r2, c1:c2);

cx = x0 - c1 + 1;
cy = y0 - r1 + 1;

%% Zona massima di ricerca

[XX, YY] = meshgrid(1:size(roi,2), 1:size(roi,1));

raggio_ricerca = 65;

zona_ricerca = ((XX - cx).^2 + (YY - cy).^2) <= raggio_ricerca^2;

%% Region growing

r_patch = 1;

rr1 = max(1, cy - r_patch);
rr2 = min(size(roi,1), cy + r_patch);
cc1 = max(1, cx - r_patch);
cc2 = min(size(roi,2), cx + r_patch);

val_seed = median(roi(rr1:rr2, cc1:cc2), 'all');

fprintf('Valore seed locale usato dal Region Growing: %.4f\n', val_seed);

lista_tol = [0.02 0.03 0.04 0.05 0.06 0.08 0.10 0.12];

area_min = 30;
area_max = round(0.55 * numel(roi));

mask_rg_best = false(size(roi));
area_best = 0;
tol_best = lista_tol(1);

for k = 1:length(lista_tol)

    tol = lista_tol(k);

    mask_simile = abs(roi - val_seed) <= tol;

    mask_simile = mask_simile & zona_ricerca;

    mask_rg = bwselect(mask_simile, cx, cy, 8);

    area_corrente = sum(mask_rg(:));

    if area_corrente >= area_min && area_corrente <= area_max
        mask_rg_best = mask_rg;
        area_best = area_corrente;
        tol_best = tol;
        break;
    end

    if area_corrente > area_best && area_corrente <= area_max
        mask_rg_best = mask_rg;
        area_best = area_corrente;
        tol_best = tol;
    end

end

if sum(mask_rg_best(:)) == 0
    error(['Nessuna regione trovata. ', ...
           'Prova a cliccare più al centro della regione oppure aumenta lista_tol.']);
end

fprintf('\nTolleranza scelta automaticamente: %.3f\n', tol_best);
fprintf('Area iniziale Region Growing: %.0f pixel\n', area_best);

%% Pulizia region growing

mask_rg_best = bwareaopen(mask_rg_best, 10);
mask_rg_best = imclose(mask_rg_best, strel('disk', 2));
mask_rg_best = imfill(mask_rg_best, 'holes');
mask_rg_best = bwselect(mask_rg_best, cx, cy, 8);
mask_rg_best = bwareafilt(mask_rg_best, 1);

%% Maschera iniziale per active contours

mask_init = mask_rg_best;

mask_init = imclose(mask_init, strel('disk', 2));
mask_init = imfill(mask_init, 'holes');
mask_init = bwselect(mask_init, cx, cy, 8);
mask_init = bwareafilt(mask_init, 1);
mask_init = mask_init & zona_ricerca;

%% Vincolo spaziale per active contours

raggio_vincolo = 12;

vincolo_ac = imdilate(mask_init, strel('disk', raggio_vincolo));
vincolo_ac = vincolo_ac & zona_ricerca;

%% Active contours Chan-Vese

num_iter = 120;

tic;

mask_ac = activecontour(roi, mask_init, num_iter, 'Chan-Vese');

t_ac = toc;

mask_ac = mask_ac & vincolo_ac;

mask_ac = bwselect(mask_ac, cx, cy, 8);
mask_ac = bwareaopen(mask_ac, 10);
mask_ac = imclose(mask_ac, strel('disk', 2));
mask_ac = imfill(mask_ac, 'holes');

if any(mask_ac(:))
    mask_ac = bwareafilt(mask_ac, 1);
end

%% Controllo active contours

area_rg = sum(mask_rg_best(:));
area_ac = sum(mask_ac(:));

fprintf('\nArea Region Growing: %.0f pixel\n', area_rg);
fprintf('Area Active Contours: %.0f pixel\n', area_ac);

fattore_massimo = 2.0;
fattore_minimo = 0.6;

if area_ac == 0

    warning('Active Contours ha prodotto una maschera vuota. Uso il Region Growing.');
    mask_roi = mask_rg_best;

elseif area_ac > fattore_massimo * area_rg

    warning('Active Contours si è allargato troppo. Uso il Region Growing.');
    mask_roi = mask_rg_best;

elseif area_ac < fattore_minimo * area_rg

    warning('Active Contours è troppo piccolo. Uso il Region Growing.');
    mask_roi = mask_rg_best;

else

    mask_roi = mask_ac;

end

%% Pulizia finale

mask_roi = bwareaopen(mask_roi, 10);
mask_roi = imclose(mask_roi, strel('disk', 1));
mask_roi = bwmorph(mask_roi, 'bridge', Inf);
mask_roi = imfill(mask_roi, 'holes');
mask_roi = bwselect(mask_roi, cx, cy, 8);
mask_roi = bwareafilt(mask_roi, 1);

%% Reinserimento nell'immagine originale

mask_finale = false(size(im_eq));

mask_finale(r1:r2, c1:c2) = mask_roi;

%% Visualizzazione evoluzione

figure;
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imagesc(zona_ricerca);
axis image off;
colormap gray;
title('Zona di ricerca');

nexttile;
imagesc(mask_rg_best);
axis image off;
title('Region Growing');

nexttile;
imagesc(mask_init);
axis image off;
title('Maschera iniziale AC');

nexttile;
imagesc(mask_ac);
axis image off;
title('Active Contours');

nexttile;
imagesc(mask_roi);
axis image off;
title('Risultato nella ROI');

nexttile;
imagesc(mask_finale);
axis image off;
title('Risultato finale');

sgtitle('Evoluzione della segmentazione ibrida migliorata');

%% Ground truth

gt_file = 'gt_brain.mat';

if isfile(gt_file)

    load(gt_file, 'gt_mask');
    fprintf('\nGround Truth caricata da %s\n', gt_file);

else

    figure;
    imshow(im_eq, []);
    title('Disegna contorno ventricolo / regione da segmentare - Ground Truth');

    h = drawpolygon('Color', 'r', 'LineWidth', 2);
    wait(h);

    gt_mask = createMask(h);

    save(gt_file, 'gt_mask');
    close;

    fprintf('\nGround Truth salvata in %s\n', gt_file);

end

gt = logical(gt_mask);

%% Metriche

margin = 5;

stats_gt = regionprops(gt, 'BoundingBox');

if isempty(stats_gt)
    error('La Ground Truth è vuota. Disegna una Ground Truth valida.');
end

bb = stats_gt(1).BoundingBox;

x1_bb = max(1, floor(bb(1) - margin));
y1_bb = max(1, floor(bb(2) - margin));
x2_bb = min(size(im_eq,2), ceil(bb(1) + bb(3) + margin));
y2_bb = min(size(im_eq,1), ceil(bb(2) + bb(4) + margin));

roi_bb = false(size(im_eq));
roi_bb(y1_bb:y2_bb, x1_bb:x2_bb) = true;

gt_roi = gt & roi_bb;
pred_roi = mask_finale & roi_bb;

dice_val = diceLocal(gt_roi, pred_roi);

iou_val = jaccard(gt_roi, pred_roi);

hd_val = hausdorffDist(gt_roi, pred_roi);

hd95_val = hausdorffDist95(gt_roi, pred_roi);

TP = sum(gt_roi(:) & pred_roi(:));
FP = sum(~gt_roi(:) & pred_roi(:));
FN = sum(gt_roi(:) & ~pred_roi(:));

prec_val = TP / (TP + FP + eps);

rec_val = TP / (TP + FN + eps);

%% Tabella riassuntiva

fprintf('\n================================================================================\n');
fprintf('  METRICHE - Segmentazione Ibrida MRI - bounding box GT, margine %d px\n', margin);
fprintf('================================================================================\n');

fprintf('%-28s | %-7s | %-7s | %-8s | %-8s | %-12s | %-8s\n', ...
    'Metodo', 'Dice', 'IoU', 'HD', 'HD95', 'Prec/Rec', 'Tempo(s)');

fprintf('%s\n', repmat('-', 1, 105));

fprintf('%-28s | %.4f  | %.4f  | %8.2f | %8.2f | %.3f/%.3f | %8.2f\n', ...
    'Ibrido MRI', dice_val, iou_val, hd_val, hd95_val, ...
    prec_val, rec_val, t_ac);

fprintf('%s\n', repmat('-', 1, 105));

%% Sovrapposizione risultato

figure;
imagesc(im_eq);
axis image;
colormap gray;
colorbar;
hold on;

overlay = cat(3, ones(size(im_eq)), zeros(size(im_eq)), zeros(size(im_eq)));
h = imagesc(overlay);

alpha(h, 0.35 * double(mask_finale));

title(sprintf('Segmentazione ibrida migliorata MRI - Dice = %.3f', dice_val));

%% Confronto GT predizione

figure;
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imagesc(gt_roi);
axis image off;
colormap gray;
title('Ground Truth locale');

nexttile;
imagesc(pred_roi);
axis image off;
title('Predizione locale');

nexttile;
imagesc(gt_roi + 2*pred_roi);
axis image off;
title('Confronto GT / Pred');

sgtitle(sprintf('Confronto locale - Dice = %.3f, IoU = %.3f', dice_val, iou_val));

%% Validazione geometrica

stats = regionprops(mask_finale, 'Area', 'Perimeter', 'Centroid', 'BoundingBox');

fprintf('\n--- VALIDAZIONE GEOMETRICA SEGMENTAZIONE ---\n');

if isempty(stats)

    fprintf('Nessuna regione segmentata.\n');

else

    fprintf('Area segmentata: %.0f pixel\n', stats.Area);
    fprintf('Perimetro: %.2f pixel\n', stats.Perimeter);
    fprintf('Centroide: x = %.2f, y = %.2f\n', ...
        stats.Centroid(1), stats.Centroid(2));
    fprintf('Bounding box: [x y larghezza altezza] = [%.2f %.2f %.2f %.2f]\n', ...
        stats.BoundingBox);

end

%% Funzioni locali

function d = diceLocal(A, B)

    A = logical(A);
    B = logical(B);

    if ~any(A(:)) && ~any(B(:))
        d = 1;
    elseif ~any(A(:)) || ~any(B(:))
        d = 0;
    else
        d = 2 * sum(A(:) & B(:)) / (sum(A(:)) + sum(B(:)));
    end

end

function hd = hausdorffDist(A, B)

    A = logical(A);
    B = logical(B);

    if ~any(A(:)) || ~any(B(:))
        hd = NaN;
        return;
    end

    d1 = bwdist(A);
    hd1 = max(d1(B));

    d2 = bwdist(B);
    hd2 = max(d2(A));

    hd = max(hd1, hd2);

end

function hd95 = hausdorffDist95(A, B)

    A = logical(A);
    B = logical(B);

    if ~any(A(:)) || ~any(B(:))
        hd95 = NaN;
        return;
    end

    d1 = bwdist(A);
    hd1 = prctile(d1(B), 95);

    d2 = bwdist(B);
    hd2 = prctile(d2(A), 95);

    hd95 = max(hd1, hd2);

end
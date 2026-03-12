%% ============================================================
%  ER_2025 - MULTICLASS EEG DECODING (MATLAB 2018b)
%
%  Three-class condition coding:
%    0 = Neutral viewing        (orig 3)
%    1 = Natural negative       (orig 2)
%    2 = Reappraisal            (orig 4)
%
%  Feature space:
%    ROI × band log10(power) extracted with Welch's method
%    from the stimulus window 0.30-3.80 s
%
%  ROIs:
%    F = Fz, FCz, FC1, FC2, F1, F2
%    C = C3, C1, Cz, C2, C4
%    P = P3, Pz, P4, POz
%    O = O1, Oz, O2
%    T = T7, T8
%
%  Bands:
%    Theta = 4-8 Hz
%    Alpha = 9-12 Hz
%    Beta  = 15-30 Hz
%
%  Classifier:
%    Multiclass linear SVM (ECOC)
%
%  Validation:
%    Repeated within-subject stratified K-fold CV
%    Majority voting across repetitions
%    Within-subject permutation test
%
%  Additional analyses:
%    A) ROI × band subset scouting
%    B) Within-subject effect sizes for top-performing subsets
%    C) ROI-level PSD visualizations
%    D) ROI-level spectrogram visualizations
%% ============================================================

clear; clc
restoredefaultpath; rehash toolboxcache

%% ===== EEGLAB =====
setpref('eeglab','nointernet',1);
eeglabPath = 'C:\Users\mdomi\Documents\Toolbox\eeglab2025.0.0';
assert(exist(fullfile(eeglabPath,'eeglab.m'),'file')==2, 'No encuentro eeglab.m');
addpath(eeglabPath);
eeglab nogui

%% ===== FieldTrip =====
ftPath = 'C:\Users\mdomi\Documents\Toolbox\fieldtrip-20230118';
assert(exist(fullfile(ftPath,'ft_defaults.m'),'file')==2, 'No encuentro ft_defaults.m');
addpath(ftPath);
ft_defaults

%% ===== Data =====
dataPath = 'F:\EEGO\ER_2025\PRE_PAPER_DERS_ML\';
files  = dir(fullfile(dataPath,'*.set'));
nSubj  = numel(files);

%% ===== Parameters =====
baselineTime = [-0.3 0];

% A priori ROIs used for ROI × band feature extraction
roiFrontal   = {'Fz','FCz','FC1','FC2','F1','F2'};
roiCentral   = {'C3','C1','Cz','C2','C4'};
roiParietal  = {'P3','Pz','P4','POz'};
roiOccipital = {'O1','Oz','O2'};
roiTemporal  = {'T7','T8'};

% Stimulus window for oscillatory power estimation
winStimPow   = [0.30 3.80];

% Frequency bands
bandTheta = [4 8];
bandAlpha = [9 12];
bandBeta  = [15 30];

% Class labels
classList = [0 1 2];   % 0=Neutral, 1=Natural negative, 2=Reappraisal

% Cross-validation
cv_mode = 'within';    % primary analysis in the paper
Kfold   = 5;
nRepCV  = 20;          % repeated CV for stability + majority voting
rng(1);

%% --- Global storage by trial ---
X = [];
y = [];
subjID = [];

%% --- Accumulators for descriptive figures ---
paper = struct();
paper.psdC = struct();
paper.psd  = struct();
paper.spec = struct();

roiNames = {'F','C','P','O','T'};
roiDef   = struct('F',{roiFrontal}, 'C',{roiCentral}, 'P',{roiParietal}, ...
                  'O',{roiOccipital}, 'T',{roiTemporal});

%% ============================================================
%  Loop by subject: load -> baseline -> trialwise features + descriptive accumulators
%% ============================================================
for s = 1:nSubj
    EEG = pop_loadset('filename', files(s).name, 'filepath', dataPath);
    ft_data = eeglab2fieldtrip(EEG,'raw');

    % Trial labels
    ti_struct = table2struct(ft_data.trialinfo);
    if isfield(ti_struct,'edftype')
        trials = table2array(ft_data.trialinfo(:, {'edftype'}));
    else
        trials = table2array(ft_data.trialinfo(:, {'type'}));
    end

    if ~strcmp(class(trials),'double')
        trials = cell2mat(trials);
        if ~isempty(trials) && trials(1) == 's'
            trials = trials(:,2);
        end
        trials = str2num(trials); %#ok<ST2NM>
    end

    % ============================================================
    %  Create 3-class variable:
    %   0 = Neutral (3)
    %   1 = Natural Negative (2)
    %   2 = Reappraise (4)
    % ============================================================
    trials3 = nan(size(trials));
    trials3(trials == 3) = 0;
    trials3(trials == 2) = 1;
    trials3(trials == 4) = 2;

    % Within-subject baseline (keeps trials)
    cfg = [];
    cfg.baseline = baselineTime;
    ft_bl = ft_timelockbaseline(cfg, ft_data);

    idxValid = find(~isnan(trials3));
    if isempty(idxValid)
        warning('Sujeto %d: sin trials válidos.', s);
        continue;
    end

    % --- Trialwise ROI × band features ---
    [Xs, ys] = extract_features_trialwise( ...
        ft_bl, trials3, idxValid, ...
        winStimPow, ...
        roiParietal, roiFrontal, roiCentral, roiOccipital, roiTemporal, ...
        bandTheta, bandAlpha, bandBeta);

    X = [X; Xs];
    y = [y; ys];
    subjID = [subjID; s*ones(size(ys))];

    fprintf('Subject %d/%d: %s -> %d trials | ROIxBand features | nFeat=%d\n', ...
        s, nSubj, files(s).name, numel(ys), size(Xs,2));

    % --- Descriptive accumulators: PSD and spectrograms ---
    paper = accumulate_paper_figures( ...
        paper, ft_bl, trials3, idxValid, winStimPow, ...
        roiDef, roiNames);
end

% Ensure categories
ycat = categorical(y, classList);

%% ============================================================
%  CV: WITHIN-SUBJECT (K-fold) or LOSO + ECOC(linear SVM)
%% ============================================================
classes = categorical(classList, classList);
t = templateSVM('KernelFunction','linear', 'Standardize',false);

% Global predictions
yhat_all = categorical(nan(size(y)), classList);

% Subject-level metrics (within)
subj_metrics = table('Size',[nSubj 7], ...
    'VariableTypes',{'double','double','double','double','double','double','double'}, ...
    'VariableNames',{'nTrials','BalAcc','MacroF1','Kappa','BalAcc_SDrep','MacroF1_SDrep','Kappa_SDrep'});

for s = 1:nSubj
    idxS = (subjID == s);
    if ~any(idxS), continue; end

    Xs = X(idxS,:);
    ys = ycat(idxS);

    if strcmpi(cv_mode,'lso')
        % --- LOSO ---
        trainIdx = ~idxS;
        testIdx  = idxS;

        Xtr = X(trainIdx,:); ytr = ycat(trainIdx);
        Xte = X(testIdx,:);

        [Xtrz, Xtez] = zscore_fold(Xtr, Xte);
        wtr = class_weights(ytr, classList);

        Mdl = fitcecoc(Xtrz, ytr, 'Learners', t, 'ClassNames', classes, 'Weights', wtr);
        ypred = predict(Mdl, Xtez);
        yhat_all(testIdx) = ypred;

    elseif strcmpi(cv_mode,'within')
        % --- WITHIN-SUBJECT: Repeated stratified K-fold (vote) + SDrep ---
        nS = sum(idxS);
        subj_metrics.nTrials(s) = nS;

        K = min(Kfold, nS);
        K = min(K, min(countcats(ys)));
        if K < 2
            warning('Sujeto %d: muy pocos trials (%d) para CV.', s, nS);
            continue;
        end

        votes  = zeros(nS, numel(classList));
        balRep = nan(nRepCV,1);
        f1Rep  = nan(nRepCV,1);
        kRep   = nan(nRepCV,1);

        for r = 1:nRepCV
            cvp = cvpartition(ys,'KFold',K);
            ypredS = categorical(nan(nS,1), classList);

            for f = 1:cvp.NumTestSets
                trF = training(cvp,f);
                teF = test(cvp,f);

                Xtr = Xs(trF,:);  ytr = ys(trF);
                Xte = Xs(teF,:);

                [Xtrz, Xtez] = zscore_fold(Xtr, Xte);
                wtr = class_weights(ytr, classList);

                Mdl = fitcecoc(Xtrz, ytr, 'Learners', t, 'ClassNames', classes, 'Weights', wtr);
                ypredS(teF) = predict(Mdl, Xtez);
            end

            % Votes
            for k = 1:numel(classList)
                cls = categorical(classList(k), classList);
                votes(:,k) = votes(:,k) + double(ypredS == cls);
            end

            % Metrics for this repetition (without vote) -> SDrep
            [balRep(r), f1Rep(r), kRep(r)] = metrics_multiclass_only(ys, ypredS, classList);
        end

        % Final prediction by majority vote
        [~, imax] = max(votes, [], 2);
        ypred_vote = categorical(classList(imax(:)), classList);
        ypred_vote = ypred_vote(:);

        [bS, fS, kS] = metrics_multiclass_only(ys(:), ypred_vote, classList);

        subj_metrics.BalAcc(s)  = bS;
        subj_metrics.MacroF1(s) = fS;
        subj_metrics.Kappa(s)   = kS;

        subj_metrics.BalAcc_SDrep(s)  = std(balRep,'omitnan');
        subj_metrics.MacroF1_SDrep(s) = std(f1Rep,'omitnan');
        subj_metrics.Kappa_SDrep(s)   = std(kRep,'omitnan');

        fprintf('   (secondary) subject %d: BalAcc mean=%.3f SDrep=%.3f | MacroF1 mean=%.3f SDrep=%.3f | Kappa mean=%.3f SDrep=%.3f\n', ...
            s, mean(balRep,'omitnan'), std(balRep,'omitnan'), ...
            mean(f1Rep,'omitnan'),  std(f1Rep,'omitnan'), ...
            mean(kRep,'omitnan'),   std(kRep,'omitnan'));

        fprintf('Within-CV (vote, rep=%d) subject %d: BalAcc=%.3f | MacroF1=%.3f | Kappa=%.3f (n=%d)\n', ...
            nRepCV, s, bS, fS, kS, nS);
    else
        error('cv_mode no soportado: %s', cv_mode);
    end
end

%% ============================================================
%  Global metrics
%% ============================================================
if strcmpi(cv_mode,'within')
    [balAcc, macroF1, kappa, yhat_voted] = run_cv_generic(X, ycat, subjID, classList, cv_mode, Kfold, nRepCV, t, classes);
    [~, ~, ~, cm, perClass] = metrics_multiclass(ycat, yhat_voted, classList);
else
    [balAcc, macroF1, kappa, cm, perClass] = metrics_multiclass(ycat, yhat_all, classList);
end

if strcmpi(cv_mode,'within')
    stats_sec = run_cv_repeated_meanmetrics(X, ycat, subjID, classList, Kfold, nRepCV, t, classes);

    fprintf('\n=== SECONDARY METRIC (no vote; mean metrics across repetitions) ===\n');
    fprintf('BalAcc mean=%.3f | mean(rep SD)=%.3f\n', stats_sec.BalAcc_mean, stats_sec.BalAcc_repSD_mean);
    fprintf('MacroF1 mean=%.3f | mean(rep SD)=%.3f\n', stats_sec.MacroF1_mean, stats_sec.MacroF1_repSD_mean);
    fprintf('Kappa  mean=%.3f | mean(rep SD)=%.3f\n', stats_sec.Kappa_mean,  stats_sec.Kappa_repSD_mean);
end

%% ============================================================
%  PERMUTATION TEST (1000) - within-subject label shuffling
%% ============================================================
assert(strcmpi(cv_mode,'within'), 'Permutation test implementado solo para cv_mode="within".');
nPerm = 1000;
nRepPerm = 1;
fprintf('\n=== PERMUTATION TEST (%d permutations) ===\n', nPerm);

balAcc_real = balAcc;
balAcc_null = zeros(nPerm,1);

for p = 1:nPerm
    yperm = ycat;

    for s = 1:max(subjID)
        idx = find(subjID == s);
        n   = numel(idx);
        if n > 1
            tmp = yperm(idx);
            yperm(idx) = tmp(randperm(n));
        end
    end

    [balAcc_null(p), ~, ~] = run_cv_generic( ...
        X, yperm, subjID, classList, cv_mode, Kfold, nRepPerm, t, classes);

    if mod(p,100)==0
        fprintf('Perm %d/%d\n', p, nPerm);
    end
end

null_mean = mean(balAcc_null);
null_sd   = std(balAcc_null);
p_value   = mean(balAcc_null >= balAcc_real);
z_score   = (balAcc_real - null_mean) / max(null_sd, eps);

fprintf('\nBalAcc real: %.4f\n', balAcc_real);
fprintf('Null mean ± SD: %.4f ± %.4f\n', null_mean, null_sd);
fprintf('z-score: %.3f\n', z_score);
fprintf('p-value (empirical): %.4f\n', p_value);

figure('Color','w','Name','Permutation Test BalAcc');
histogram(balAcc_null,30); hold on;
xline(balAcc_real,'r','LineWidth',2);
xlabel('Balanced Accuracy'); ylabel('Count');
title(sprintf('Permutation Test (%d) - within-subject', nPerm));

fprintf('\n=== MULTICLASS DECODING RESULTS (%s | ROIxBand | ECOC+Linear SVM) ===\n', cv_mode);
fprintf('Balanced Accuracy: %.3f\n', balAcc);
fprintf('Macro-F1: %.3f\n', macroF1);
fprintf('Cohen''s Kappa: %.3f\n', kappa);
disp('Confusion matrix (rows=true, cols=pred):'); disp(cm);
disp('Per-class metrics:'); disp(perClass);

try
    figure('Color','w','Name',sprintf('Confusion Matrix (%s | ROIxBand)',cv_mode));
    confusionchart(cm, string(classList));
catch ME
    warning('confusionchart falló: %s', ME.message);
end

if strcmpi(cv_mode,'within')
    validS = ~isnan(subj_metrics.BalAcc);
    fprintf('\n=== SUBJECT-LEVEL SUMMARY (within) ===\n');
    fprintf('Mean BalAcc: %.3f (SD=%.3f)\n', mean(subj_metrics.BalAcc(validS)), std(subj_metrics.BalAcc(validS)));
    fprintf('Mean MacroF1: %.3f (SD=%.3f)\n', mean(subj_metrics.MacroF1(validS)), std(subj_metrics.MacroF1(validS)));
    fprintf('Mean Kappa: %.3f (SD=%.3f)\n', mean(subj_metrics.Kappa(validS)), std(subj_metrics.Kappa(validS)));
end

%% ============================================================
%  (A) ROI × BAND SUBSETS (same CV) + sorted tables in console
%% ============================================================
fprintf('\n=== FEATURE SCOUTING: ROI × BAND SUBSETS (cv=%s) ===\n', cv_mode);
subsetTable = run_feature_scouting(X, ycat, subjID, classList, cv_mode, Kfold, nRepCV, t, classes);
disp(subsetTable);

%% ============================================================
%  (A1) EFFECT SIZE ROI × BAND (Hedges g + Bootstrap CI)
%  (A2) FDR correction
%% ============================================================
if exist('subsetTable','var') && ~isempty(subsetTable)

    fprintf('\n=== EFFECT SIZE ROI × BAND (Hedges g + Bootstrap CI) ===\n');

    kTop = 3;
    kTop = min(kTop, height(subsetTable));
    topSets = subsetTable.Set(1:kTop);

    pvals = zeros(numel(topSets),1);
    gvals = zeros(numel(topSets),1);
    CI_low = zeros(numel(topSets),1);
    CI_high = zeros(numel(topSets),1);

    for i = 1:numel(topSets)
        setName = char(topSets(i));
        cols = get_feature_columns(setName, size(X,2));

        subjVals = zeros(max(subjID),3);

        for s = 1:max(subjID)
            idxS = subjID==s;
            for c = 0:2
                idxC = idxS & (ycat == categorical(c, classList));
                subjVals(s,c+1) = mean(mean(X(idxC,cols),2));
            end
        end

        % Paper-relevant comparison: Natural negative vs Reappraisal
        x1 = subjVals(:,2);
        x2 = subjVals(:,3);

        [g, ci_low, ci_high, p] = hedges_gz_bootstrap(x1,x2,1000);

        gvals(i) = g;
        CI_low(i) = ci_low;
        CI_high(i) = ci_high;
        pvals(i) = p;

        fprintf('%s | Reappraisal - NatNeg: g=%.3f | CI=[%.3f %.3f] | p=%.4f\n', ...
            setName, g, ci_low, ci_high, p);
    end

    % -------- FDR --------
    [p_sorted, sort_idx] = sort(pvals);
    m = numel(pvals);
    p_fdr_sorted = p_sorted .* m ./ (1:m)';
    p_fdr_sorted = min(1, p_fdr_sorted);

    for i = m-1:-1:1
        p_fdr_sorted(i) = min(p_fdr_sorted(i), p_fdr_sorted(i+1));
    end

    p_fdr = zeros(size(pvals));
    p_fdr(sort_idx) = p_fdr_sorted;

    fprintf('\n--- FDR corrected ---\n');
    for i = 1:numel(topSets)
        fprintf('%s | p_FDR=%.4f\n', char(topSets(i)), p_fdr(i));
    end
end

%% ============================================================
%  (C) PSD 1-40 Hz by ROI (overlay)
%  (D) Spectrogram 1-40 Hz by ROI (grid)
%  (C2) PSD by condition (all ROIs + F/O)
%% ============================================================
try
    plot_roi_psd_overlay(paper, [1 40]);
    plot_roi_spectrogram_grid(paper, [1 40]);
    plot_psd_by_condition_allROIs(paper, [1 40]);
    plot_psd_by_condition_FO(paper, [1 40]);
catch ME
    warning('PSD/Spectrogram falló: %s', ME.message);
end

%% ======================= LOCAL FUNCTIONS =======================

function [Xtrz, Xtez] = zscore_fold(Xtr, Xte)
    mu = mean(Xtr,1);
    sd = std(Xtr,[],1); sd(sd==0) = 1;
    Xtrz = (Xtr - mu) ./ sd;
    Xtez = (Xte - mu) ./ sd;
end

function subsetTable = run_feature_scouting(X, ycat, subjID, classList, cv_mode, Kfold, nRepCV, t, classes)
    % Column order in X:
    % [F_theta F_alpha F_beta, C_theta C_alpha C_beta,
    %  P_theta P_alpha P_beta, O_theta O_alpha O_beta,
    %  (T_theta T_alpha T_beta optional)]

    nFeat = size(X,2);
    hasT = (nFeat == 15);

    idx = struct();
    idx.F.theta=1; idx.F.alpha=2; idx.F.beta=3;
    idx.C.theta=4; idx.C.alpha=5; idx.C.beta=6;
    idx.P.theta=7; idx.P.alpha=8; idx.P.beta=9;
    idx.O.theta=10; idx.O.alpha=11; idx.O.beta=12;
    if hasT
        idx.T.theta=13; idx.T.alpha=14; idx.T.beta=15;
    end

    sets = {};
    sets(end+1,:) = {'ALL', 1:nFeat};

    theta = [idx.F.theta idx.C.theta idx.P.theta idx.O.theta]; if hasT, theta=[theta idx.T.theta]; end
    alpha = [idx.F.alpha idx.C.alpha idx.P.alpha idx.O.alpha]; if hasT, alpha=[alpha idx.T.alpha]; end
    beta  = [idx.F.beta  idx.C.beta  idx.P.beta  idx.O.beta ]; if hasT, beta =[beta  idx.T.beta ]; end

    sets(end+1,:) = {'theta_allROIs', theta};
    sets(end+1,:) = {'alpha_allROIs', alpha};
    sets(end+1,:) = {'beta_allROIs',  beta};

    sets(end+1,:) = {'F_allBands', [idx.F.theta idx.F.alpha idx.F.beta]};
    sets(end+1,:) = {'C_allBands', [idx.C.theta idx.C.alpha idx.C.beta]};
    sets(end+1,:) = {'P_allBands', [idx.P.theta idx.P.alpha idx.P.beta]};
    sets(end+1,:) = {'O_allBands', [idx.O.theta idx.O.alpha idx.O.beta]};
    if hasT
        sets(end+1,:) = {'T_allBands', [idx.T.theta idx.T.alpha idx.T.beta]};
    end

    sets(end+1,:) = {'F_theta', idx.F.theta};
    sets(end+1,:) = {'F_alpha', idx.F.alpha};
    sets(end+1,:) = {'F_beta',  idx.F.beta};
    sets(end+1,:) = {'P_theta', idx.P.theta};
    sets(end+1,:) = {'P_alpha', idx.P.alpha};
    sets(end+1,:) = {'P_beta',  idx.P.beta};

    subsetTable = table('Size',[size(sets,1) 5], ...
        'VariableTypes',{'string','double','double','double','double'}, ...
        'VariableNames',{'Set','nFeat','BalAcc','MacroF1','Kappa'});

    for i = 1:size(sets,1)
        name = string(sets{i,1});
        cols = sets{i,2};
        Xt = X(:, cols);

        [balAcc, macroF1, kappa, ~] = run_cv_generic(Xt, ycat, subjID, classList, cv_mode, Kfold, nRepCV, t, classes);

        subsetTable.Set(i) = name;
        subsetTable.nFeat(i) = numel(cols);
        subsetTable.BalAcc(i) = balAcc;
        subsetTable.MacroF1(i) = macroF1;
        subsetTable.Kappa(i) = kappa;

        fprintf('  %-14s | nFeat=%2d | BalAcc=%.3f | MacroF1=%.3f | Kappa=%.3f\n', ...
            name, numel(cols), balAcc, macroF1, kappa);
    end

    subsetTable = sortrows(subsetTable, {'BalAcc','MacroF1'}, {'descend','descend'});
    fprintf('\n=== SORTED TABLE (desc BalAcc, desc MacroF1) ===\n');
end

function [balAcc, macroF1, kappa, yhat_all] = run_cv_generic(X, ycat, subjID, classList, cv_mode, Kfold, nRepCV, t, classes)
    nSubj = max(subjID);
    yhat_all = categorical(nan(size(ycat)), classList);

    for s = 1:nSubj
        idxS = (subjID == s);
        if ~any(idxS), continue; end

        Xs = X(idxS,:);
        ys = ycat(idxS);

        if strcmpi(cv_mode,'lso')
            trainIdx = ~idxS;
            testIdx  = idxS;

            Xtr = X(trainIdx,:); ytr = ycat(trainIdx);
            Xte = X(testIdx,:);

            [Xtrz, Xtez] = zscore_fold(Xtr, Xte);
            wtr = class_weights(ytr, classList);

            Mdl = fitcecoc(Xtrz, ytr, 'Learners', t, 'ClassNames', classes, 'Weights', wtr);
            yhat_all(testIdx) = predict(Mdl, Xtez);

        elseif strcmpi(cv_mode,'within')
            nS = sum(idxS);
            K = min(Kfold, nS);
            K = min(K, min(countcats(ys)));
            if K < 2
                continue;
            end

            votes = zeros(nS, numel(classList));

            for r = 1:nRepCV
                cvp = cvpartition(ys,'KFold',K);
                ypredS = categorical(nan(nS,1), classList);

                for f = 1:cvp.NumTestSets
                    trF = training(cvp,f);
                    teF = test(cvp,f);

                    Xtr = Xs(trF,:); ytr = ys(trF);
                    Xte = Xs(teF,:);

                    [Xtrz, Xtez] = zscore_fold(Xtr, Xte);
                    wtr = class_weights(ytr, classList);

                    Mdl = fitcecoc(Xtrz, ytr, 'Learners', t, 'ClassNames', classes, 'Weights', wtr);
                    ypredS(teF) = predict(Mdl, Xtez);
                end

                for k = 1:numel(classList)
                    cls = categorical(classList(k), classList);
                    votes(:,k) = votes(:,k) + double(ypredS == cls);
                end
            end

            [~, imax] = max(votes, [], 2);
            tmp_pred = categorical(classList(imax(:)), classList);
            yhat_all(idxS) = tmp_pred(:);
        else
            error('cv_mode no soportado: %s', cv_mode);
        end
    end

    [balAcc, macroF1, kappa] = metrics_multiclass_only(ycat, yhat_all, classList);
end

function paper = accumulate_paper_figures(paper, ft_bl, trials3, idxTrials, winStim, roiDef, roiNames)
    % Accumulates:
    %  (1) PSD: ROI average across all conditions
    %  (2) PSD by condition for all available ROIs (paper.psdC)
    %  (3) Spectrogram: ROI average across all conditions

    fs = ft_bl.fsample;

    if ~isfield(paper,'psd') || isempty(paper.psd),  paper.psd  = struct(); end
    if ~isfield(paper,'spec')|| isempty(paper.spec), paper.spec = struct(); end
    if ~isfield(paper,'psdC')|| isempty(paper.psdC), paper.psdC = struct(); end

    fRange = [1 40];

    win   = max(32, round(0.50*fs));
    nover = max(0,  round(0.50*win));
    nfft  = max(256, 2^nextpow2(win));

    winS   = max(64, round(0.50*fs));
    noverS = max(0,  round(0.90*winS));
    nfftS  = max(256, 2^nextpow2(winS));

    for ii = 1:numel(idxTrials)
        tr = idxTrials(ii);

        cls = trials3(tr);
        if isnan(cls) || ~ismember(cls,[0 1 2])
            continue;
        end
        k = cls + 1;

        dat  = ft_bl.trial{tr};
        tvec = ft_bl.time{tr};

        mStim = (tvec>=winStim(1) & tvec<=winStim(2));
        if ~any(mStim)
            continue;
        end

        seg = dat(:, mStim);

        for r = 1:numel(roiNames)
            rName = roiNames{r};

            if ~isfield(roiDef, rName)
                continue;
            end

            idxR = match_str(ft_bl.label, roiDef.(rName));
            if isempty(idxR)
                continue;
            end

            roiSig = mean(seg(idxR,:), 1);

            % ---- PSD (Welch) ----
            [Pxx, F] = pwelch(double(roiSig(:)), win, nover, nfft, fs);
            mF = (F>=fRange(1) & F<=fRange(2));
            F1 = F(mF);
            P1 = Pxx(mF);

            if ~isfield(paper.psd, rName) || ~isfield(paper.psd.(rName),'sumPxx') || isempty(paper.psd.(rName).sumPxx)
                paper.psd.(rName).sumPxx = zeros(size(P1));
                paper.psd.(rName).n      = 0;
                paper.psd.(rName).F      = F1;
            end
            paper.psd.(rName).sumPxx = paper.psd.(rName).sumPxx + P1(:);
            paper.psd.(rName).n      = paper.psd.(rName).n + 1;

            if ~isfield(paper.psdC, rName) || ~isfield(paper.psdC.(rName),'sumPxx') || isempty(paper.psdC.(rName).sumPxx)
                paper.psdC.(rName).sumPxx = zeros(numel(F1), 3);
                paper.psdC.(rName).n      = zeros(1,3);
                paper.psdC.(rName).F      = F1;
            end
            paper.psdC.(rName).sumPxx(:,k) = paper.psdC.(rName).sumPxx(:,k) + P1(:);
            paper.psdC.(rName).n(1,k)      = paper.psdC.(rName).n(1,k) + 1;

            % ---- Spectrogram ----
            [S, Fsp, Tsp] = spectrogram(double(roiSig(:)), winS, noverS, nfftS, fs);
            Spow = abs(S).^2;

            mFs  = (Fsp>=fRange(1) & Fsp<=fRange(2));
            Fsp1 = Fsp(mFs);
            Sp1  = Spow(mFs,:);

            if ~isfield(paper.spec, rName) || ~isfield(paper.spec.(rName),'sumS') || isempty(paper.spec.(rName).sumS)
                paper.spec.(rName).sumS = zeros(size(Sp1), 'double');
                paper.spec.(rName).n    = 0;
                paper.spec.(rName).F    = Fsp1;
                paper.spec.(rName).T    = Tsp;
            end
            paper.spec.(rName).sumS = paper.spec.(rName).sumS + Sp1;
            paper.spec.(rName).n    = paper.spec.(rName).n + 1;
        end
    end
end

function plot_roi_psd_overlay(paper, fRange)
    roiOrder = {'F','C','P','O','T'};

    figure('Color','w','Name','ROI PSD (Welch) overlay 1-40 Hz');
    hold on;

    h = gobjects(0);
    leg = {};

    for i = 1:numel(roiOrder)
        r = roiOrder{i};
        if ~isfield(paper.psd, r) || paper.psd.(r).n == 0
            continue;
        end

        F = paper.psd.(r).F;
        P = paper.psd.(r).sumPxx ./ paper.psd.(r).n;

        hh = plot(F, 10*log10(P+eps), 'LineWidth', 1.5);
        h(end+1) = hh; %#ok<AGROW>
        leg{end+1} = r; %#ok<AGROW>
    end

    if isempty(h)
        warning('PSD overlay: no hay ROIs con datos.');
        return;
    end

    xlim(fRange);
    grid on;
    xlabel('Hz');
    ylabel('Power (dB)');
    title(sprintf('ROI-averaged PSD (%d-%d Hz)', fRange(1), fRange(2)));
    legend(h, leg, 'Location','southwest', 'Box','off');
    hold off;
end

function plot_roi_spectrogram_grid(paper, fRange)
    roiOrder = {'F','C','P','O','T'};
    nR = numel(roiOrder);

    figure('Color','w','Name','ROI Spectrogram 1-40 Hz');
    for i = 1:nR
        r = roiOrder{i};

        if ~isfield(paper.spec, r) || paper.spec.(r).n == 0
            subplot(2,3,i); axis off; title(sprintf('ROI %s (no data)', r));
            continue;
        end

        F = paper.spec.(r).F;
        T = paper.spec.(r).T;
        S = paper.spec.(r).sumS ./ paper.spec.(r).n;

        subplot(2,3,i);
        imagesc(T, F, 10*log10(S+eps));
        axis xy;
        ylim(fRange);
        xlabel('Time (s)');
        ylabel('Hz');
        title(sprintf('ROI %s | Spectrogram', r));
        colorbar;
    end
    subplot(2,3,6); axis off;
end

function [Xs, ys] = extract_features_trialwise( ...
    ft_bl, trials, idxTrials, ...
    winStimPow, ...
    roiParietal, roiFrontal, roiCentral, roiOccipital, roiTemporal, ...
    bandTheta, bandAlpha, bandBeta)

    fs = ft_bl.fsample;
    nT = numel(idxTrials);

    idxP = match_str(ft_bl.label, roiParietal);
    idxF = match_str(ft_bl.label, roiFrontal);
    idxC = match_str(ft_bl.label, roiCentral);
    idxO = match_str(ft_bl.label, roiOccipital);
    idxT = match_str(ft_bl.label, roiTemporal);

    if isempty(idxP), error('Parietal ROI not found.'); end
    if isempty(idxF), error('Frontal ROI not found.'); end
    if isempty(idxC), error('Central ROI not found.'); end
    if isempty(idxO), error('Occipital ROI not found.'); end

    useTemporal = ~isempty(idxT);
    nFeat = 12 + (useTemporal * 3);
    Xs = zeros(nT, nFeat);
    ys = zeros(nT, 1);

    for i = 1:nT
        tr = idxTrials(i);
        ys(i) = trials(tr);

        dat  = ft_bl.trial{tr};
        tvec = ft_bl.time{tr};

        pSig = mean(dat(idxP,:), 1);
        fSig = mean(dat(idxF,:), 1);
        cSig = mean(dat(idxC,:), 1);
        oSig = mean(dat(idxO,:), 1);
        if useTemporal
            tSig = mean(dat(idxT,:), 1);
        end

        mStim = (tvec >= winStimPow(1) & tvec <= winStimPow(2));
        if ~any(mStim)
            mStim = (tvec >= winStimPow(1) & tvec <= max(tvec));
        end

        pWin = pSig(mStim);
        fWin = fSig(mStim);
        cWin = cSig(mStim);
        oWin = oSig(mStim);

        thF = log10(bandpower_welch(fWin, fs, bandTheta) + eps);
        alF = log10(bandpower_welch(fWin, fs, bandAlpha) + eps);
        beF = log10(bandpower_welch(fWin, fs, bandBeta ) + eps);

        thC = log10(bandpower_welch(cWin, fs, bandTheta) + eps);
        alC = log10(bandpower_welch(cWin, fs, bandAlpha) + eps);
        beC = log10(bandpower_welch(cWin, fs, bandBeta ) + eps);

        thP = log10(bandpower_welch(pWin, fs, bandTheta) + eps);
        alP = log10(bandpower_welch(pWin, fs, bandAlpha) + eps);
        beP = log10(bandpower_welch(pWin, fs, bandBeta ) + eps);

        thO = log10(bandpower_welch(oWin, fs, bandTheta) + eps);
        alO = log10(bandpower_welch(oWin, fs, bandAlpha) + eps);
        beO = log10(bandpower_welch(oWin, fs, bandBeta ) + eps);

        featVec = [thF alF beF, thC alC beC, thP alP beP, thO alO beO];

        if useTemporal
            tWin = tSig(mStim);
            thT = log10(bandpower_welch(tWin, fs, bandTheta) + eps);
            alT = log10(bandpower_welch(tWin, fs, bandAlpha) + eps);
            beT = log10(bandpower_welch(tWin, fs, bandBeta ) + eps);
            featVec = [featVec, thT alT beT];
        end

        Xs(i,:) = featVec;
    end
end

function bp = bandpower_welch(x, fs, bandHz)
    x = double(x(:));
    n = numel(x);
    win = min(round(0.5*fs), n);
    if win < 32, win = min(n, 32); end
    nover = round(0.5*win);
    nfft = max(256, 2^nextpow2(win));
    [Pxx, F] = pwelch(x, win, nover, nfft, fs);
    m = (F>=bandHz(1) & F<=bandHz(2));
    bp = trapz(F(m), Pxx(m));
end

function w = class_weights(ycat, classList)
    w = ones(numel(ycat),1);
    for k = 1:numel(classList)
        cls = categorical(classList(k), classList);
        nk = sum(ycat == cls);
        if nk > 0
            w(ycat == cls) = numel(ycat) / (numel(classList) * nk);
        end
    end
end

function [balAcc, macroF1, kappa, cm, perClass] = metrics_multiclass(ytrue, ypred, classList)
    classes = categorical(classList, classList);

    ytrue = ytrue(:);
    ypred = ypred(:);

    cm = confusionmat(ytrue, ypred, 'Order', classes);

    K = numel(classList);
    prec = zeros(K,1); rec = zeros(K,1); f1 = zeros(K,1); sup = zeros(K,1);

    for k = 1:K
        tp = cm(k,k);
        fn = sum(cm(k,:)) - tp;
        fp = sum(cm(:,k)) - tp;
        sup(k) = sum(cm(k,:));

        rec(k)  = tp / max(tp+fn,1);
        prec(k) = tp / max(tp+fp,1);
        f1(k)   = 2*(prec(k)*rec(k)) / max(prec(k)+rec(k), eps);
    end

    balAcc = mean(rec);
    macroF1 = mean(f1);

    N = sum(cm(:));
    po = trace(cm)/max(N,1);
    rowMarg = sum(cm,2);
    colMarg = sum(cm,1);
    pe = sum(rowMarg .* colMarg') / max(N^2,1);
    kappa = (po - pe) / max(1 - pe, eps);

    perClass = table(classList(:), sup, prec, rec, f1, ...
        'VariableNames', {'Class','Support','Precision','Recall','F1'});
end

function [balAcc, macroF1, kappa] = metrics_multiclass_only(ytrue, ypred, classList)
    ytrue = ytrue(:);
    ypred = ypred(:);

    classes = categorical(classList, classList);

    ok = ~ismissing(ytrue) & ~ismissing(ypred);
    cm = confusionmat(ytrue(ok), ypred(ok), 'Order', classes);

    K = numel(classList);
    prec = zeros(K,1); rec = zeros(K,1); f1 = zeros(K,1);

    for k = 1:K
        tp = cm(k,k);
        fn = sum(cm(k,:)) - tp;
        fp = sum(cm(:,k)) - tp;

        rec(k)  = tp / max(tp+fn,1);
        prec(k) = tp / max(tp+fp,1);
        f1(k)   = 2*(prec(k)*rec(k)) / max(prec(k)+rec(k), eps);
    end

    balAcc  = mean(rec);
    macroF1 = mean(f1);

    N = sum(cm(:));
    po = trace(cm)/max(N,1);
    rowMarg = sum(cm,2);
    colMarg = sum(cm,1);
    pe = sum(rowMarg .* colMarg') / max(N^2,1);
    kappa = (po - pe) / max(1 - pe, eps);
end

function cols = get_feature_columns(setName, nFeat)
    hasT = (nFeat==15);

    idxF = [1 2 3];
    idxC = [4 5 6];
    idxP = [7 8 9];
    idxO = [10 11 12];
    if hasT, idxT = [13 14 15]; end

    theta = [1 4 7 10]; if hasT, theta = [theta 13]; end
    alpha = [2 5 8 11]; if hasT, alpha = [alpha 14]; end
    beta  = [3 6 9 12]; if hasT, beta  = [beta  15]; end

    switch char(setName)
        case 'ALL'
            cols = 1:nFeat;

        case 'theta_allROIs'
            cols = theta;
        case 'alpha_allROIs'
            cols = alpha;
        case 'beta_allROIs'
            cols = beta;

        case 'F_allBands'
            cols = idxF;
        case 'C_allBands'
            cols = idxC;
        case 'P_allBands'
            cols = idxP;
        case 'O_allBands'
            cols = idxO;
        case 'T_allBands'
            if hasT, cols = idxT; else, cols = []; end

        case 'F_theta'
            cols = 1;
        case 'F_alpha'
            cols = 2;
        case 'F_beta'
            cols = 3;

        case 'P_theta'
            cols = 7;
        case 'P_alpha'
            cols = 8;
        case 'P_beta'
            cols = 9;

        otherwise
            warning('Set no reconocido: %s. Usando ALL.', char(setName));
            cols = 1:nFeat;
    end
end

function [gz, ci_low, ci_high, p] = hedges_gz_bootstrap(x1, x2, nBoot)
    x1 = x1(:); x2 = x2(:);
    ok = ~isnan(x1) & ~isnan(x2);
    x1 = x1(ok); x2 = x2(ok);

    d = x2 - x1;
    n = numel(d);

    if n < 3
        gz = NaN; ci_low = NaN; ci_high = NaN; p = NaN;
        return;
    end

    dz = mean(d) / std(d, 0);
    J  = 1 - (3/(4*n - 9));
    gz = J * dz;

    gboot = zeros(nBoot,1);
    for b = 1:nBoot
        db = d(randi(n, n, 1));
        dzb = mean(db) / std(db, 0);
        Jb  = 1 - (3/(4*n - 9));
        gboot(b) = Jb * dzb;
    end

    ci_low  = prctile(gboot, 2.5);
    ci_high = prctile(gboot,97.5);

    p = 2*min(mean(gboot <= 0), mean(gboot >= 0));
    p = min(p, 1);
end

function plot_psd_by_condition_FO(paper, fRange)
    condNames = {'Neutral','NatNeg','Reapp'};
    roiOrder  = {'F','O'};

    axFS = 14;
    tlFS = 16;
    lgFS = 12;

    for iR = 1:numel(roiOrder)
        r = roiOrder{iR};

        if ~isfield(paper,'psdC') || ~isfield(paper.psdC, r) || ~isfield(paper.psdC.(r),'sumPxx')
            warning('PSD cond: ROI %s no tiene datos en paper.psdC.', r);
            continue;
        end

        F = paper.psdC.(r).F;
        S = paper.psdC.(r).sumPxx;
        n = paper.psdC.(r).n;

        figure('Color','w','Name',sprintf('PSD by condition | ROI %s', r));
        hold on;

        for k = 1:3
            if n(k)==0, continue; end
            P = S(:,k) ./ n(k);
            plot(F, 10*log10(P+eps), 'LineWidth', 1.8);
        end

        xlim(fRange);
        grid on;
        xlabel('Frequency (Hz)', 'FontSize', axFS);
        ylabel('Power (dB)', 'FontSize', axFS);
        title(sprintf('ROI %s: PSD by condition (%d-%d Hz)', r, fRange(1), fRange(2)), 'FontSize', tlFS);

        legend(condNames, 'Location','southwest', 'Box','off', 'FontSize', lgFS);
        set(gca, 'FontSize', axFS);
        hold off;
    end
end

function plot_psd_by_condition_allROIs(paper, fRange)
    condNames = {'Neutral','NatNeg','Reapp'};
    roiOrder  = {'F','C','P','O','T'};

    figure('Color','w','Name','PSD by condition | All ROIs');

    nPlot = 0;
    for iR = 1:numel(roiOrder)
        r = roiOrder{iR};

        if ~isfield(paper,'psdC') || ~isfield(paper.psdC, r) || ~isfield(paper.psdC.(r),'sumPxx') || isempty(paper.psdC.(r).sumPxx)
            continue;
        end

        n = paper.psdC.(r).n;
        if all(n==0), continue; end
        nPlot = nPlot + 1;
    end

    if nPlot==0
        warning('PSD cond: no hay datos en paper.psdC para ningún ROI.');
        return;
    end

    nCols = 3;
    nRows = ceil(nPlot/nCols);

    pIdx = 0;
    for iR = 1:numel(roiOrder)
        r = roiOrder{iR};

        if ~isfield(paper,'psdC') || ~isfield(paper.psdC, r) || ~isfield(paper.psdC.(r),'sumPxx') || isempty(paper.psdC.(r).sumPxx)
            continue;
        end

        F = paper.psdC.(r).F;
        S = paper.psdC.(r).sumPxx;
        n = paper.psdC.(r).n;

        if all(n==0), continue; end

        pIdx = pIdx + 1;
        subplot(nRows, nCols, pIdx); hold on;

        for k = 1:3
            if n(k)==0, continue; end
            P = S(:,k) ./ n(k);
            plot(F, 10*log10(P+eps), 'LineWidth', 1.6);
        end

        xlim(fRange);
        grid on;
        title(sprintf('ROI %s', r));
        xlabel('Hz');
        ylabel('Power (dB)');

        if pIdx==1
            legend(condNames, 'Location','southwest', 'Box','off');
        end

        hold off;
    end
end

function stats = run_cv_repeated_meanmetrics(X, ycat, subjID, classList, Kfold, nRepCV, t, classes)
    nSubj = max(subjID);

    bal_all = nan(nSubj,1);
    bal_sd  = nan(nSubj,1);

    f1_all  = nan(nSubj,1);
    f1_sd   = nan(nSubj,1);

    kap_all = nan(nSubj,1);
    kap_sd  = nan(nSubj,1);

    for s = 1:nSubj
        idxS = (subjID == s);
        if ~any(idxS), continue; end

        Xs = X(idxS,:);
        ys = ycat(idxS);

        nS = sum(idxS);
        K = min(Kfold, nS);
        K = min(K, min(countcats(ys)));
        if K < 2, continue; end

        balRep = nan(nRepCV,1);
        f1Rep  = nan(nRepCV,1);
        kRep   = nan(nRepCV,1);

        for r = 1:nRepCV
            cvp = cvpartition(ys,'KFold',K);
            ypredS = categorical(nan(nS,1), classList);

            for f = 1:cvp.NumTestSets
                trF = training(cvp,f);
                teF = test(cvp,f);

                Xtr = Xs(trF,:);  ytr = ys(trF);
                Xte = Xs(teF,:);

                [Xtrz, Xtez] = zscore_fold(Xtr, Xte);
                wtr = class_weights(ytr, classList);

                Mdl = fitcecoc(Xtrz, ytr, 'Learners', t, 'ClassNames', classes, 'Weights', wtr);
                ypredS(teF) = predict(Mdl, Xtez);
            end

            [balRep(r), f1Rep(r), kRep(r)] = metrics_multiclass_only(ys, ypredS, classList);
        end

        bal_all(s) = mean(balRep,'omitnan');  bal_sd(s) = std(balRep,'omitnan');
        f1_all(s)  = mean(f1Rep,'omitnan');   f1_sd(s)  = std(f1Rep,'omitnan');
        kap_all(s) = mean(kRep,'omitnan');    kap_sd(s) = std(kRep,'omitnan');
    end

    ok = ~isnan(bal_all);

    stats.BalAcc_mean = mean(bal_all(ok));
    stats.BalAcc_sd   = std(bal_all(ok));
    stats.BalAcc_repSD_mean = mean(bal_sd(ok));

    stats.MacroF1_mean = mean(f1_all(ok));
    stats.MacroF1_sd   = std(f1_all(ok));
    stats.MacroF1_repSD_mean = mean(f1_sd(ok));

    stats.Kappa_mean = mean(kap_all(ok));
    stats.Kappa_sd   = std(kap_all(ok));
    stats.Kappa_repSD_mean = mean(kap_sd(ok));
end
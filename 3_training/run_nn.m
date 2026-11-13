function run_nn(cfgFile, nThreads)
% Train the classification network under different settings, one per row of cfgFile (CSV), and export each
% trained network for evaluation in Python.
%
% The network and its training are those of the first PeatCover version:
%   patternnet, tansig hidden layer(s) + softmax output, trainscg, cross-entropy, inputs scaled by
%   removeconstantrows + mapminmax, early stopping on a separate validation set (max_fail 15), at most 800 epochs.
% Only the column(s) named in the CSV change between runs:
%   name     run name
%   feats    which inputs: 17 (first version) or 17iv (17 + OPERA inundated vegetation = the 18 inputs of the revised map)
%   hidden   hidden layer sizes, e.g. 10 (paper) or 32_16 (two layers)
%   ntrain   number of training pixels (0 = all 7 M training pixels of the paper's split)
%   ratio    non-peat per peat pixel in training (1 = the paper's 1:1)
%   fold     -1 = the paper's random train/validation split; 0..4 = spatial block left out (cross-validation)
%   seed     random seed of the subsampling and of the weight initialisation
% Optional columns:
%   epochs   maximum number of epochs (default 800 = the paper)
%   foldvar  'fold5' = the 5 x 5 deg tile folds of 5_evaluation/02_cv_folds.py (WORK/cv5/labels_fold5.mat)
% Data: WORK/training_table.mat (4_mapping/01_training_table.py). Output: WORK/nn/<name>.mat (exported weights)
% + results/tables/nn_runs.csv. WORK = environment variable PEATCOVER_WORK.
% Usage: run_nn('networks_final.csv')
if nargin < 2, nThreads = 12; end
maxNumCompThreads(nThreads);
work = getenv('PEATCOVER_WORK');
outDir = fullfile(work, 'nn'); if ~exist(outDir, 'dir'), mkdir(outDir); end
here = fileparts(mfilename('fullpath'));
logFile = fullfile(here, '..', 'results', 'tables', 'nn_runs.csv');
if ~exist(fileparts(logFile), 'dir'), mkdir(fileparts(logFile)); end

opts = detectImportOptions(cfgFile, 'Delimiter', ',');
txt = intersect({'name', 'feats', 'hidden', 'foldvar', 'data'}, opts.VariableNames);
opts = setvartype(opts, txt, 'string');                          % '17iv' and hidden sizes such as 32_16 stay text
cfg = readtable(cfgFile, opts);
hasCol = @(n) ismember(n, cfg.Properties.VariableNames);
loaded = ''; F5 = [];
% columns of D.X: 1-17 = the paper's inputs (9 = WTD), 18 = OPERA inundated vegetation, 19 = OPERA open water
featSets = struct('f17', 1:17, 'f19', 1:19, 'f17iv', [1:17 18], 'f17ow', [1:17 19], 'f16', [1:8 10:17], 'f18', [1:8 10:19]);

for r = 1:height(cfg)
    c = cfg(r, :);
    name = char(c.name);
    lockFile = fullfile(outDir, [name '.lock']);       % several MATLAB queues may share a configuration
    if exist(fullfile(outDir, [name '.mat']), 'file') || exist(lockFile, 'file')
        fprintf('%s exists or is being trained, skip\n', name); continue
    end
    fclose(fopen(lockFile, 'w'));
    dataFile = 'training_table.mat';
    if hasCol('data') && ~ismissing(c.data) && strlength(c.data) > 0, dataFile = char(c.data); end
    if ~strcmp(dataFile, loaded)
        D = load(fullfile(work, dataFile), 'X', 'y', 'split', 'fold');
        y = double(D.y(:)); split = D.split(:); loaded = dataFile;
    end
    fold = double(D.fold(:));
    if hasCol('foldvar') && ~ismissing(c.foldvar) && c.foldvar == "fold5"
        if isempty(F5), F5 = load(fullfile(work, 'cv5', 'labels_fold5.mat')); end
        fold = double(F5.fold5(:));
    end
    maxEpochs = 800;
    if hasCol('epochs') && ~isnan(c.epochs) && c.epochs > 0, maxEpochs = c.epochs; end
    rng(c.seed);
    cols = featSets.(['f' char(c.feats)]);
    hidden = str2num(strrep(char(c.hidden), '_', ' ')); %#ok<ST2NM>
    % ---- training / validation pools
    if c.fold < 0
        trPool = find(split == 0); vaPool = find(split == 1);
    else
        pool = find(fold ~= c.fold);
        pool = pool(randperm(numel(pool)));
        nva = round(0.15 * numel(pool));
        vaPool = pool(1:nva); trPool = pool(nva + 1:end);
    end
    trInd = pick(trPool, y, c.ntrain, c.ratio);
    vaInd = pick(vaPool, y, round(0.2 * c.ntrain), c.ratio);
    X = D.X([trInd; vaInd], cols)';
    T = [y([trInd; vaInd])' == 0; y([trInd; vaInd])' == 1];
    ok = all(isfinite(X), 1);
    X = X(:, ok); T = T(:, ok);
    nTr = sum(ok(1:numel(trInd)));
    % ---- network and training settings of the paper
    net = patternnet(hidden);
    net.inputs{1}.processFcns = {'removeconstantrows', 'mapminmax'};
    net.divideFcn = 'divideind'; net.divideMode = 'sample';
    net.divideParam.trainInd = 1:nTr;
    net.divideParam.valInd = nTr + 1:size(X, 2);
    net.divideParam.testInd = [];
    net.trainFcn = 'trainscg'; net.performFcn = 'crossentropy';
    net.trainParam.epochs = maxEpochs; net.trainParam.max_fail = 15; net.trainParam.min_grad = 1e-6;
    net.trainParam.sigma = 5.0e-5; net.trainParam.lambda = 5.0e-7; net.trainParam.goal = 0;
    net.trainParam.showWindow = false; net.trainParam.showCommandLine = false;
    t0 = tic;
    [net, tr] = train(net, X, T);
    secs = toc(t0);
    export_net(net, fullfile(outDir, [name '.mat']));
    delete(lockFile);
    fid = fopen(logFile, 'a');
    fprintf(fid, '%s,%s,%s,%d,%g,%d,%d,%d,%d,%.5f,%.0f,%s\n', name, char(c.feats), regexprep(num2str(hidden), '\s+', '_'), c.ntrain, c.ratio, ...
            c.fold, c.seed, nTr, tr.best_epoch, tr.best_vperf, secs, datestr(now, 'yyyy-mm-dd HH:MM'));
    fclose(fid);
    fprintf('%s: %d train px, best epoch %d, val cross-entropy %.4f, %.0f s\n', name, nTr, tr.best_epoch, tr.best_vperf, secs);
end
end


function ind = pick(pool, y, n, ratio)
% n pixels from pool with ratio non-peat per peat pixel (n = 0: the whole pool, as in the paper)
if n <= 0
    ind = pool; return
end
p = pool(y(pool) == 1); q = pool(y(pool) == 0);
np = round(n / (1 + ratio)); nq = n - np;
ind = [p(randperm(numel(p), min(np, numel(p)))); q(randperm(numel(q), min(nq, numel(q))))];
end

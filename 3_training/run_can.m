function run_can(cfgFile, nWorkers)
% Train the Canadian regression network (4_mapping/05_canada_regression_vb.py), one run per row of cfgFile (CSV: name,
% feats, seed), and export each network for Python. Usage: run_can('networks_canada.csv')
%
% Network and training as in the first PeatCover version:
%   patternnet(10) (tansig hidden layer; with one target row MATLAB makes the output layer logsig and scales the target with
%   mapminmax), inputs removeconstantrows + mapminmax, trainlm, mse, 50 epochs, other parameters at their defaults,
%   trained in parallel ('useParallel').
% The only difference: the paper drew the 5 % training / 1 % validation pixels with dividerand from the balanced sample of
% 193.5 M pixels; 05_canada_regression_vb.py draws the same numbers directly (WORK/canada/can_samples.mat, split 0 / 1), so divideind is used.
%   feats   17 = the paper's inputs, 17iv = + OPERA inundated vegetation (column 18)
% Output: WORK/canada/nn/<name>.mat (exported weights and input / output scaling) + nn/can_runs.csv
if nargin < 2, nWorkers = 12; end
work = fullfile(getenv('PEATCOVER_WORK'), 'canada');
outDir = fullfile(work, 'nn'); if ~exist(outDir, 'dir'), mkdir(outDir); end
opts = detectImportOptions(cfgFile, 'Delimiter', ',');
opts = setvartype(opts, {'name', 'feats'}, 'string');
cfg = readtable(cfgFile, opts);
D = [];
for r = 1:height(cfg)
    c = cfg(r, :);
    name = char(c.name);
    lockFile = fullfile(outDir, [name '.lock']);
    if exist(fullfile(outDir, [name '.mat']), 'file') || exist(lockFile, 'file')
        fprintf('%s exists or is being trained, skip\n', name); continue
    end
    fclose(fopen(lockFile, 'w'));
    if isempty(D)
        D = load(fullfile(work, 'can_samples.mat'), 'X', 'y', 'split');
        D.split = D.split(:); D.y = double(D.y(:));
    end
    cols = 1:17;
    if c.feats == "17iv", cols = 1:18; end
    rng(c.seed);
    tr0 = find(D.split == 0); va0 = find(D.split == 1);
    X = double(D.X([tr0; va0], cols)');
    Y = D.y([tr0; va0])';
    ok = all(isfinite(X), 1);
    nTr = sum(ok(1:numel(tr0)));
    X = X(:, ok); Y = Y(ok);
    net = patternnet(10);
    net.inputs{1}.processFcns = {'removeconstantrows', 'mapminmax'};
    net.divideFcn = 'divideind'; net.divideMode = 'sample';
    net.divideParam.trainInd = 1:nTr;
    net.divideParam.valInd = nTr + 1:size(X, 2);
    net.divideParam.testInd = [];
    net.trainFcn = 'trainlm'; net.performFcn = 'mse';
    net.trainParam.epochs = 50;
    net.trainParam.showWindow = false; net.trainParam.showCommandLine = true; net.trainParam.show = 5;
    if isempty(gcp('nocreate'))
        [~, q] = fileparts(cfgFile);
        js = fullfile(outDir, ['jobs_' q]);                           % own folder per MATLAB session (two queues in parallel)
        if ~exist(js, 'dir'), mkdir(js); end
        cl = parcluster('local'); cl.NumWorkers = nWorkers; cl.JobStorageLocation = js;
        parpool(cl, nWorkers);
    end
    t0 = tic;
    [net, tr] = train(net, X, Y, 'useParallel', 'yes');
    secs = toc(t0);
    export_net(net, fullfile(outDir, [name '.mat']));
    delete(lockFile);
    fid = fopen(fullfile(outDir, 'can_runs.csv'), 'a');
    fprintf(fid, '%s,%s,%d,%d,%d,%d,%.3f,%.3f,%s,%s,%.0f,%s\n', name, char(c.feats), c.seed, nTr, size(X, 2) - nTr, tr.best_epoch, ...
            tr.best_perf, tr.best_vperf, net.layers{2}.transferFcn, strjoin(net.outputs{2}.processFcns, '+'), secs, ...
            datestr(now, 'yyyy-mm-dd HH:MM'));
    fclose(fid);
    fprintf('%s: %d train px, best epoch %d, train mse %.2f, val mse %.2f, output %s, %.0f s\n', name, nTr, tr.best_epoch, ...
            tr.best_perf, tr.best_vperf, net.layers{2}.transferFcn, secs);
end
end

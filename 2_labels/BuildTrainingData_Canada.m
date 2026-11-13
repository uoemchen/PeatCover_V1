%% Canada: training data of the regression network (peat fraction of the PEATMAP polygons)
% Target: water-corrected peat fraction (%) of the Peatlands of Canada polygon of each pixel, 0 < fraction <= 100
%         (<tile>_frac_corrected.tif, 1_inputs/MERIT_DP_Canada.m).
% Inputs: MERIT (8, MERIT_Canada), WTD (1), Landsat 8 July bands 1-7 and NDVI = 17 inputs, as BuildTrainingData.m.
% Pixels that are permanent water (permWater.mat), with Landsat values outside 0-10000 or with a missing input are removed.
% Output: Training_database/canada_training_data/<tile>_trainingData.mat (input, target, latS, lonS)
% Next: 3_training/run_can.m

clear; clc;

root = getenv('PEATCOVER_ROOT');
outputBaseFolder  = fullfile(root, 'Training_database/canada_training_data');
landsatFolder     = fullfile(root, 'Processed/1_Landsat8/Landsat8_NA_July');
MERITBaseFolder   = fullfile(root, 'Processed/3_MERIT/MERIT_Canada');
WTDBaseFolder     = fullfile(root, 'Processed/4_WTD/WTD_NA');
peatmapBaseFolder = fullfile(root, 'Processed/2_PEATMAP/Canada/Tiles');
if ~exist(outputBaseFolder, 'dir'); mkdir(outputBaseFolder); end

nVar = 8 + 1 + 7 + 1;
meritVars = {'dist0005', 'dist0100', 'dist1000', 'hand0005', 'hand0100', 'hand1000', 'slope', 'elevation'};
landsatCols = 10:16;
redCol = 13;                    % band 4
nirCol = 14;                    % band 5

landsatFiles = dir(fullfile(landsatFolder, '*.mat'));
for i = 1:length(landsatFiles)
    [~, regionName, ~] = fileparts(landsatFiles(i).name);

    fileRasterFrac = fullfile(peatmapBaseFolder, regionName, [regionName '_frac_corrected.tif']);
    if ~isfile(fileRasterFrac)
        continue;
    end
    fprintf('PROCESSING %s\n', regionName);
    [peatmapLR, R] = readgeoraster(fileRasterFrac);
    listInd = find(peatmapLR > 0 & peatmapLR <= 100);
    if isempty(listInd)
        continue;
    end
    input = nan(length(listInd), nVar, 'single');

    % MERIT (columns 1-8)
    for j = 1:length(meritVars)
        v = loadVar(MERITBaseFolder, regionName, meritVars{j});
        input(:, j) = v(listInd);
    end

    % WTD (column 9)
    try
        v = loadVar(WTDBaseFolder, regionName, 'wtd');
        input(:, 9) = v(listInd);
    catch
        fprintf('Missing WTD for %s\n', regionName);
        continue;
    end

    % permanent water
    try
        permStruct = load(fullfile(MERITBaseFolder, regionName, 'permWater.mat'));
        permMask = permStruct.permWater(listInd);
    catch
        permMask = zeros(size(listInd));
    end

    % Landsat 8 bands 1-7 (columns 10-16)
    filePath = fullfile(landsatFolder, [regionName '.mat']);
    allL8 = loadVar(filePath, '', 'landsat');
    for b = 1:7
        band = reshape(allL8(:, b), R.RasterSize);
        input(:, 9 + b) = band(listInd);
    end

    % pixel centres
    dlat = R.CellExtentInLatitude;
    dlon = R.CellExtentInLongitude;
    latVec = single(linspace(R.LatitudeLimits(2)-dlat/2, R.LatitudeLimits(1)+dlat/2, R.RasterSize(1)));
    lonVec = single(linspace(R.LongitudeLimits(1)+dlon/2, R.LongitudeLimits(2)-dlon/2, R.RasterSize(2)));
    [lonGrid, latGrid] = meshgrid(lonVec, latVec);
    lonS = lonGrid(listInd);
    latS = latGrid(listInd);

    % remove invalid Landsat values and permanent water
    toBeRemoved = any(input(:, landsatCols) < 0 | input(:, landsatCols) > 10000, 2) | (permMask ~= 0);
    input(toBeRemoved, :) = [];
    target = peatmapLR(listInd);
    target(toBeRemoved) = [];
    lonS(toBeRemoved) = [];
    latS(toBeRemoved) = [];

    % reflectance, NDVI (column 17)
    input(:, landsatCols) = input(:, landsatCols) / 10000;
    input(:, 17) = (input(:, nirCol) - input(:, redCol)) ./ (input(:, nirCol) + input(:, redCol));

    % remove pixels with any missing input
    nanRows = any(isnan(input), 2);
    input(nanRows, :) = [];
    target(nanRows) = [];
    lonS(nanRows) = [];
    latS(nanRows) = [];

    fileOut = fullfile(outputBaseFolder, [regionName '_trainingData.mat']);
    save(fileOut, 'input', 'target', 'latS', 'lonS', '-v7.3');
    fprintf('SAVED %s (%d pixels)\n', fileOut, size(input, 1));
end
fprintf('ALL DONE\n');

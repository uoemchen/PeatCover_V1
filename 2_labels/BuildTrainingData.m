%% Training data of each tile (Europe and the United States): the 17 inputs at every labelled pixel
% Labels: peat_labels.py (1 = peat, 0 = non-peat; -1 is not used).
% Inputs (1_inputs): MERIT (8), WTD (1), Landsat 8 bands 1-7 (reflectance x 10000) and NDVI, on the Landsat tile grid.
% Pixels with Landsat values outside 0-10000 or with any missing input are removed.
% Output: <outputBaseFolder>/<tile>_trainingData.mat with
%   input  (n x 17, single): dist0005 dist0100 dist1000 hand0005 hand0100 hand1000 slope elevation wtd b1..b7 ndvi
%   target (n x 1): 1 = peat, 0 = non-peat; latS, lonS: pixel centres
% Next: sample_split_labels.py

clear; clc;

root = getenv('PEATCOVER_ROOT');
labelFolder      = fullfile(root, 'Training_database/label_tiles');
outputBaseFolder = fullfile(root, 'Training_database/training_data_tiles');
regions = {'Europe', 'NA'};      % Landsat, MERIT and WTD folders: Landsat8_<region>_July, MERIT_<region>, WTD_<region>

if ~exist(outputBaseFolder, 'dir'); mkdir(outputBaseFolder); end
nVar = 8 + 1 + 7 + 1;           % MERIT (8) + WTD (1) + Landsat (7) + NDVI (1)
meritVars = {'dist0005', 'dist0100', 'dist1000', 'hand0005', 'hand0100', 'hand1000', 'slope', 'elevation'};
landsatCols = 10:16;
redCol = 13;                    % band 4
nirCol = 14;                    % band 5

labelFiles = dir(fullfile(labelFolder, '*_labels.tif'));
tileNames = strrep({labelFiles.name}, '_labels.tif', '');

if isempty(gcp('nocreate'))
    parpool('local', 8);
end

for r = 1:numel(regions)
    landsatFolder   = fullfile(root, 'Processed/1_Landsat8', ['Landsat8_' regions{r} '_July']);
    MERITBaseFolder = fullfile(root, 'Processed/3_MERIT', ['MERIT_' regions{r}]);
    WTDBaseFolder   = fullfile(root, 'Processed/4_WTD', ['WTD_' regions{r}]);

    parfor t = 1:length(tileNames)
        tileName = tileNames{t};
        labelFile = fullfile(labelFolder, [tileName '_labels.tif']);
        landsatFile = fullfile(landsatFolder, [tileName '.mat']);
        if ~exist(landsatFile, 'file')
            continue;           % tile of the other region
        end
        try
            labels = readgeoraster(labelFile);
            landsatTemp = load(landsatFile, 'R');
            R = landsatTemp.R;

            validIndices = find(labels == 0 | labels == 1);
            if isempty(validIndices)
                continue;
            end
            nSamples = length(validIndices);
            target = labels(validIndices);
            input = nan(nSamples, nVar, 'single');

            % pixel centres
            dlat = R.CellExtentInLatitude;
            dlon = R.CellExtentInLongitude;
            lat_full = single(linspace(R.LatitudeLimits(2) - dlat/2, R.LatitudeLimits(1) + dlat/2, R.RasterSize(1)));
            lon_full = single(linspace(R.LongitudeLimits(1) + dlon/2, R.LongitudeLimits(2) - dlon/2, R.RasterSize(2)));
            [lon_grid, lat_grid] = meshgrid(lon_full, lat_full);
            latS = lat_grid(validIndices);
            lonS = lon_grid(validIndices);

            % MERIT (columns 1-8); a missing variable stays NaN and its pixels are removed below
            for j = 1:length(meritVars)
                try
                    v = loadVar(MERITBaseFolder, tileName, meritVars{j});
                    input(:, j) = v(validIndices);
                catch ME
                    fprintf('%s: %s missing (%s)\n', tileName, meritVars{j}, ME.message);
                end
            end

            % WTD (column 9)
            try
                v = loadVar(WTDBaseFolder, tileName, 'wtd');
                input(:, 9) = v(validIndices);
            catch ME
                fprintf('%s: wtd missing (%s)\n', tileName, ME.message);
            end

            % Landsat 8 bands 1-7 (columns 10-16)
            try
                landsatData = load(landsatFile);
                for k = 1:7
                    band = reshape(landsatData.allL8(:, k), R.RasterSize);
                    input(:, 9 + k) = band(validIndices);
                end
            catch ME
                fprintf('%s: Landsat error (%s)\n', tileName, ME.message);
            end

            % remove invalid Landsat values, convert to reflectance, NDVI (column 17)
            invalidRows = any(input(:, landsatCols) < 0 | input(:, landsatCols) > 10000, 2);
            input(invalidRows, :) = [];
            target(invalidRows) = [];
            latS(invalidRows) = [];
            lonS(invalidRows) = [];
            input(:, landsatCols) = input(:, landsatCols) / 10000;
            input(:, 17) = (input(:, nirCol) - input(:, redCol)) ./ (input(:, nirCol) + input(:, redCol));

            % remove pixels with any missing input
            nanRows = any(isnan(input), 2);
            input(nanRows, :) = [];
            target(nanRows) = [];
            latS(nanRows) = [];
            lonS(nanRows) = [];

            outputFile = fullfile(outputBaseFolder, [tileName '_trainingData.mat']);
            parsave(outputFile, input, target, latS, lonS);
            fprintf('%s: %d samples (%d peat) -> %s\n', tileName, size(input, 1), sum(target == 1), outputFile);
        catch ME
            fprintf('%s failed: %s\n', tileName, ME.message);
        end
    end
end


function parsave(filename, input, target, latS, lonS)
    save(filename, 'input', 'target', 'latS', 'lonS', '-v7.3');
end

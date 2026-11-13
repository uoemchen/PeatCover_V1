%% Prediction stacks: the 17 inputs of every pixel of a tile
% For every Landsat tile: 8 MERIT inputs, WTD, 7 Landsat 8 bands (reflectance = value / 10000) and NDVI of the July
% composite, in this order, for all 6040 x 6040 pixels (column-major), with the pixel-centre coordinates.
% Output: <outputBaseFolder>/<tile>_predictionData.mat (input: pixels x 17, R, lonS, latS)
% Regions: Europe, NA and Dateline (MERIT inputs of the Dateline tiles: MERIT_Europe).
clear; clc;

root = getenv('PEATCOVER_ROOT');
regions = {'Europe', 'Europe'; 'NA', 'NA'; 'Dateline', 'Europe'};    % region, MERIT folder
ndviMonth = 'July';
landsatMonths = {'July'};

if isempty(gcp('nocreate'))
    parpool('local', 10);
end

for r = 1:size(regions, 1)
    outputBaseFolder = fullfile(root, 'Prediction_Database', regions{r, 1});
    if ~exist(outputBaseFolder, 'dir'); mkdir(outputBaseFolder); end
    landsatFolders = {fullfile(root, 'Processed/1_Landsat8', ['Landsat8_' regions{r, 1} '_July'])};
    MERITBaseFolder = fullfile(root, 'Processed/3_MERIT', ['MERIT_' regions{r, 2}]);
    WTDBaseFolder = fullfile(root, 'Processed/4_WTD', ['WTD_' regions{r, 1}]);

    landsatFiles = dir(fullfile(landsatFolders{1}, '*.mat'));
    numRegions = length(landsatFiles);
    fprintf('Processing %d tiles\n', numRegions);

    parfor i = 1:numRegions
        try
            [~, regionName, ~] = fileparts(landsatFiles(i).name);
            fprintf('PROCESSING %s\n', regionName);

            % raster reference of the tile
            landsat_data = load(fullfile(landsatFolders{1}, landsatFiles(i).name), 'R');
            R = landsat_data.R;

            nVar = 8 + 1 + 7 * length(landsatMonths) + 1;
            nRows = R.RasterSize(1);
            nCols = R.RasterSize(2);
            nPixels = nRows * nCols;

            % pixel-centre coordinates
            dlat = R.CellExtentInLatitude;
            dlon = R.CellExtentInLongitude;
            lat = single(linspace(R.LatitudeLimits(2) - dlat/2, R.LatitudeLimits(1) + dlat/2, R.RasterSize(1)));
            lon = single(linspace(R.LongitudeLimits(1) + dlon/2, R.LongitudeLimits(2) - dlon/2, R.RasterSize(2)));
            [lon, lat] = meshgrid(lon, lat);
            lonS = lon(:);
            latS = lat(:);

            input = nan(nPixels, nVar, 'single');
            iCol = 0;

            %% MERIT variables
            meritVars = {'dist0005', 'dist0100', 'dist1000', 'hand0005', 'hand0100', 'hand1000', 'slope', 'elevation'};
            for j = 1:length(meritVars)
                try
                    var = loadVar(MERITBaseFolder, regionName, meritVars{j});
                    iCol = iCol + 1;
                    input(:, iCol) = var(:);
                catch ME
                    warning('Missing MERIT %s for %s: %s', meritVars{j}, regionName, ME.message);
                    continue;
                end
            end

            %% WTD
            try
                var = loadVar(WTDBaseFolder, regionName, 'wtd');
                iCol = iCol + 1;
                input(:, iCol) = var(:);
            catch ME
                warning('Missing WTD for %s: %s', regionName, ME.message);
                continue;
            end

            %% Landsat 8
            ndviBand4Col = -1;
            ndviBand5Col = -1;
            for m = 1:length(landsatMonths)
                folder = landsatFolders{m};
                filePath = fullfile(folder, [regionName '.mat']);
                if ~isfile(filePath)
                    warning('Missing Landsat: %s', filePath);
                    continue;
                end
                try
                    allL8 = loadVar(filePath, '', 'landsat');
                    for b = 1:7
                        iCol = iCol + 1;
                        band = reshape(allL8(:, b), [], 1);
                        input(:, iCol) = band;
                        if strcmp(landsatMonths{m}, ndviMonth)
                            if b == 4; ndviBand4Col = iCol; end
                            if b == 5; ndviBand5Col = iCol; end
                        end
                    end
                catch ME
                    warning('Error loading Landsat month %s for %s: %s', landsatMonths{m}, regionName, ME.message);
                end
            end

            %% Landsat values -> reflectance
            landsatStartCol = 8 + 1 + 1;
            landsatEndCol = landsatStartCol + 7 * length(landsatMonths) - 1;
            min_value = min(min(input(:, landsatStartCol:landsatEndCol)));
            max_value = max(max(input(:, landsatStartCol:landsatEndCol)));
            fprintf('%s - Landsat values range: %f to %f\n', regionName, min_value, max_value);
            input(:, landsatStartCol:landsatEndCol) = input(:, landsatStartCol:landsatEndCol) / 10000;

            %% NDVI
            if ndviBand4Col > 0 && ndviBand5Col > 0
                iCol = iCol + 1;
                input(:, iCol) = (input(:, ndviBand5Col) - input(:, ndviBand4Col)) ./ ...
                                (input(:, ndviBand5Col) + input(:, ndviBand4Col));
            else
                warning('Could not compute NDVI for %s', regionName);
                continue;
            end

            fileOut = fullfile(outputBaseFolder, [regionName '_predictionData.mat']);
            parsave(fileOut, input, R, lonS, latS);
            fprintf('%s - Saved successfully\n', regionName);

        catch ME
            warning('Error processing region %s: %s\n%s', regionName, ME.message, getReport(ME));
        end
    end
end

fprintf('\nAll processing completed.\n');

function parsave(filename, input, R, lonS, latS)
    % save inside parfor
    save(filename, 'input', 'R', 'lonS', 'latS', '-v7.3');
end

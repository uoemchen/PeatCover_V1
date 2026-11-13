%% Water table depth preprocessing on the Landsat tile grid
% Annual mean water table depth of Fan et al. (2013) (waterTableDepth.m), interpolated to the pixel centres of every
% Landsat tile; pixels outside the WTD land mask are NaN.
% Output: <processedDataBaseFolder>/<tile>/wtd.mat (wtd + R)
% Regions: Europe (WTD file EURASIA), NA and Dateline (WTD file NAMERICA).
% WTDfolder: folder of <areaWTD>_WTD_annualmean.nc.

clear
clc

root = getenv('PEATCOVER_ROOT');
WTDfolder = getenv('WTD_DIR');
regions = {'Europe', 'EURASIA'; 'NA', 'NAMERICA'; 'Dateline', 'NAMERICA'};    % region, WTD file

for r = 1:size(regions, 1)
    landsatFolder = fullfile(root, 'Processed/1_Landsat8', ['Landsat8_' regions{r, 1} '_July']);
    processedDataBaseFolder = fullfile(root, 'Processed/4_WTD', ['WTD_' regions{r, 1}]);
    areaWTD = regions{r, 2};

    landsatFiles = dir(fullfile(landsatFolder, '*.mat'));

    for i = 1:length(landsatFiles)
        landsatFilePath = fullfile(landsatFiles(i).folder, landsatFiles(i).name);
        fprintf('PROCESSING %s\n', landsatFiles(i).name);

        load(landsatFilePath, 'R', 'info');

        [~, regionName, ~] = fileparts(landsatFiles(i).name);
        processedDataFolder = fullfile(processedDataBaseFolder, regionName);
        if ~exist(processedDataFolder, 'dir')
            mkdir(processedDataFolder);
        end

        myFile = fullfile(processedDataFolder, 'wtd.mat');
        fprintf('Processing Water Table Depth for %s\n', regionName);

        try
            minLat = min(R.LatitudeLimits);
            maxLat = max(R.LatitudeLimits);
            minLon = min(R.LongitudeLimits);
            maxLon = max(R.LongitudeLimits);
            res = R.CellExtentInLatitude;

            % original WTD grid
            [wtd, lat, lon, mask] = waterTableDepth(WTDfolder, minLat, maxLat, minLon, maxLon, areaWTD, 0);
            [lon, lat] = meshgrid(lon, lat);

            % pixel centres of the Landsat tile
            newLat = linspace(maxLat - res / 2, minLat + res / 2, R.RasterSize(1));
            newLon = linspace(minLon + res / 2, maxLon - res / 2, R.RasterSize(2));
            [newLon, newLat] = meshgrid(newLon, newLat);

            wtd = interp2(lon, lat, wtd, newLon, newLat);
            mask = interp2(lon, lat, single(mask), newLon, newLat);
            mask = int8(round(mask));
            wtd(~mask) = NaN;

            save(myFile, 'wtd', 'R', '-v7.3');
            fprintf('SAVED %s\n', myFile);
        catch ME
            % tiles outside the WTD product: all NaN
            fprintf('Failed to process WTD for %s: %s\n', regionName, ME.message);
            wtd = zeros(R.RasterSize) * NaN;
            save(myFile, 'wtd', 'R', '-v7.3');
        end

        fprintf('Completed processing for %s\n', regionName);
    end
end

fprintf('ALL DONE\n');

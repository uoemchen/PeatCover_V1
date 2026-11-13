%% MERIT DEM / MERIT Hydro preprocessing on the Landsat tile grid
% For every Landsat tile (.mat file of LS8_DP_multiple.m), the eight MERIT inputs are cut from the 5 x 5 deg MERIT tiles
% (windowCell.m, HydroCell.m) and interpolated to the pixel centres of the Landsat tile:
%   dist0005, dist0100, dist1000   distance to the drainage network (small, medium, large rivers)
%   hand0005, hand0100, hand1000   height above the nearest drainage
%   slope, elevation
% Output: <processedDataBaseFolder>/<tile>/<variable>.mat (variable + R)
% Regions: Europe (incl. the Dateline tiles) and NA.
% MERITfolder: folder of the MERIT tiles, with the layout expected by HydroCell.m.

clear
clc

root = getenv('PEATCOVER_ROOT');
MERITfolder = getenv('MERIT_DIR');
regions = {'Europe', 'NA'};

for r = 1:numel(regions)
    landsatFolder = fullfile(root, 'Processed/1_Landsat8', ['Landsat8_' regions{r} '_July']);
    processedDataBaseFolder = fullfile(root, 'Processed/3_MERIT', ['MERIT_' regions{r}]);

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

        % pixel centres of the Landsat tile
        minLat = min(R.LatitudeLimits);
        maxLat = max(R.LatitudeLimits);
        minLon = min(R.LongitudeLimits);
        maxLon = max(R.LongitudeLimits);
        res = R.CellExtentInLatitude;

        newLat = linspace(maxLat - res / 2, minLat + res / 2, R.RasterSize(1));
        newLon = linspace(minLon + res / 2, maxLon - res / 2, R.RasterSize(2));
        [newLon, newLat] = meshgrid(newLon, newLat);

        %% DISTANCE TO SMALL DRAINAGE
        myFile = fullfile(processedDataFolder, 'dist0005.mat');
        fprintf('Processing dist0005\n');
        try
            [dist0005, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'dest_0005');
            dist0005 = interp2(lon, lat, dist0005, newLon, newLat);
            save(myFile, 'dist0005', 'R', '-v7.3');
            fprintf('SAVED %s\n', myFile);
        catch ME
            fprintf('Failed to process dist0005 for %s: %s\n', regionName, ME.message);
        end

        %% DISTANCE TO MEDIUM DRAINAGE
        myFile = fullfile(processedDataFolder, 'dist0100.mat');
        fprintf('Processing dist0100\n');
        try
            [dist0100, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'dest_0100');
            dist0100 = interp2(lon, lat, dist0100, newLon, newLat);
            save(myFile, 'dist0100', 'R', '-v7.3');
            fprintf('SAVED %s\n', myFile);
        catch ME
            fprintf('Failed to process dist0100 for %s: %s\n', regionName, ME.message);
        end

        %% DISTANCE TO LARGE DRAINAGE
        myFile = fullfile(processedDataFolder, 'dist1000.mat');
        fprintf('Processing dist1000\n');
        try
            [dist1000, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'dest_1000');
            dist1000 = interp2(lon, lat, dist1000, newLon, newLat);
            save(myFile, 'dist1000', 'R', '-v7.3');
            fprintf('SAVED %s\n', myFile);
        catch ME
            fprintf('Failed to process dist1000 for %s: %s\n', regionName, ME.message);
        end

        %% HEIGHT ABOVE SMALL DRAINAGE
        myFile = fullfile(processedDataFolder, 'hand0005.mat');
        fprintf('Processing hand0005\n');
        try
            [hand0005, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'hand_0005');
            hand0005 = interp2(lon, lat, hand0005, newLon, newLat);
            save(myFile, 'hand0005', 'R', '-v7.3');
            fprintf('SAVED %s\n', myFile);
        catch ME
            fprintf('Failed to process hand0005 for %s: %s\n', regionName, ME.message);
        end

        %% HEIGHT ABOVE MEDIUM DRAINAGE
        myFile = fullfile(processedDataFolder, 'hand0100.mat');
        fprintf('Processing hand0100\n');
        try
            [hand0100, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'hand_0100');
            hand0100 = interp2(lon, lat, hand0100, newLon, newLat);
            save(myFile, 'hand0100', 'R', '-v7.3');
            fprintf('SAVED %s\n', myFile);
        catch ME
            fprintf('Failed to process hand0100 for %s: %s\n', regionName, ME.message);
        end

        %% HEIGHT ABOVE LARGE DRAINAGE
        myFile = fullfile(processedDataFolder, 'hand1000.mat');
        fprintf('Processing hand1000\n');
        try
            [hand1000, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'hand_1000');
            hand1000 = interp2(lon, lat, hand1000, newLon, newLat);
            save(myFile, 'hand1000', 'R', '-v7.3');
            fprintf('SAVED %s\n', myFile);
        catch ME
            fprintf('Failed to process hand1000 for %s: %s\n', regionName, ME.message);
        end

        %% SLOPE
        myFile = fullfile(processedDataFolder, 'slope.mat');
        fprintf('Processing slope\n');
        try
            [slope, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'slope');
            slope = interp2(lon, lat, slope, newLon, newLat);
            save(myFile, 'slope', 'R', '-v7.3');
            fprintf('SAVED %s\n', myFile);
        catch ME
            fprintf('Failed to process slope for %s: %s\n', regionName, ME.message);
        end

        %% ELEVATION
        myFile = fullfile(processedDataFolder, 'elevation.mat');
        fprintf('Processing elevation\n');
        try
            [elevation, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'elevation');
            elevation = interp2(lon, lat, elevation, newLon, newLat);
            save(myFile, 'elevation', 'R', '-v7.3');
            fprintf('SAVED %s\n', myFile);
        catch ME
            fprintf('Failed to process elevation for %s: %s\n', regionName, ME.message);
        end

        fprintf('Completed processing for %s\n', regionName);
    end
end

fprintf('ALL DONE\n');

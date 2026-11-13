%% Canada: MERIT inputs, permanent water and water-corrected PEATMAP peat fraction on the PEATMAP Canada tiles
% For every PEATMAP Canada tile folder (made by 2_labels/PEATMAP_DP_Canada.m from the Peatlands of Canada polygons:
% <tile>_georef.tif = polygon index, <tile>_frac.tif = peat fraction, <tile>_latlon.shp = polygons):
%   1. permanent water from the MERIT Hydro river width ('wth': -1 or > 0 = water), on the tile grid -> permWater.mat
%   2. peat fraction of each polygon relative to its land area: PEAT_PER / (100 - % of permanent water) x 100, at most
%      100 -> <tile>_frac_corrected.tif and <tile>_latlon_corrected.shp. The polygon downscaling places the peat area of a
%      polygon on its land pixels only, so the fraction is expressed per land pixel.
%   3. the eight MERIT inputs on the tile grid (as MERIT_DP_multiple.m)
% Output: Processed/3_MERIT/MERIT_Canada/<tile>/*.mat; corrected fractions next to the PEATMAP files.
% The corrected fractions are the targets of the Canadian regression network (2_labels/BuildTrainingData_Canada.m) and the
% polygon fractions of the polygon downscaling (4_mapping).

clear
clc

root = getenv('PEATCOVER_ROOT');
peatmapFolder           = fullfile(root, 'Processed/2_PEATMAP/Canada/Tiles/');
processedDataBaseFolder = fullfile(root, 'Processed/3_MERIT/MERIT_Canada');
MERITfolder             = getenv('MERIT_DIR');

d = dir(peatmapFolder);
isSubDir = [d.isdir] & ~ismember({d.name}, {'.','..'});
tileDirs = d(isSubDir);

fprintf('Found %d PEATMAP tiles in %s\n', numel(tileDirs), peatmapFolder);

for i = 1:numel(tileDirs)
    regionName = tileDirs(i).name;
    fprintf('\n=============================================\n');
    fprintf('PROCESSING TILE %s\n', regionName);

    peatmapRegionFolder = fullfile(peatmapFolder, regionName);

    processedDataFolder = fullfile(processedDataBaseFolder, regionName);
    if ~exist(processedDataFolder, 'dir')
        mkdir(processedDataFolder);
    end

    %% Tile grid from the PEATMAP polygon index raster
    fileGeoRef  = fullfile(peatmapRegionFolder, [regionName '_georef.tif']);
    if ~exist(fileGeoRef, 'file')
        fprintf('GeoRef file missing for %s, skipping this tile.\n', regionName);
        continue;
    end

    [~, R] = readgeoraster(fileGeoRef);
    minLat = min(R.LatitudeLimits);
    maxLat = max(R.LatitudeLimits);
    minLon = min(R.LongitudeLimits);
    maxLon = max(R.LongitudeLimits);
    res    = R.CellExtentInLatitude;

    newLat = linspace(maxLat - res / 2, minLat + res / 2, R.RasterSize(1));
    newLon = linspace(minLon + res / 2, maxLon - res / 2, R.RasterSize(2));
    [newLon, newLat] = meshgrid(newLon, newLat);

    %% Permanent water from the MERIT Hydro river width
    filePermWater = fullfile(processedDataFolder, 'permWater.mat');
    hasWater = false;

    try
        [permWater, latW, lonW] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'wth');

        if isempty(permWater) || all(isnan(permWater(:)))
            fprintf('No valid MERIT width (wth) for %s - will NOT correct peat fraction.\n', regionName);
            hasWater = false;
        else
            hasWater = true;
            permWater(permWater == -1 | permWater > 0) = 1;
            permWater = round(interp2(lonW, latW, permWater, newLon, newLat));
            save(filePermWater, 'permWater', 'R', '-v7.3');
            fprintf('SAVED %s\n', filePermWater);
        end

    catch ME
        fprintf('Failed to process Permanent Water for %s: %s\n', regionName, ME.message);
        hasWater = false;
    end

    %% Peat fraction relative to the land area of each polygon
    fileRasterFrac  = fullfile(peatmapRegionFolder, [regionName '_frac.tif']);
    fileRasterFracC = fullfile(peatmapRegionFolder, [regionName '_frac_corrected.tif']);
    filePeatShp     = fullfile(peatmapRegionFolder, [regionName '_latlon.shp']);
    filePeatShpC    = fullfile(peatmapRegionFolder, [regionName '_latlon_corrected.shp']);

    doFracCorrection = hasWater ...
        && exist(fileRasterFrac, 'file') ...
        && exist(filePeatShp, 'file');

    if doFracCorrection
        fprintf('Correcting peat fraction by excluding permanent water for %s\n', regionName);

        [fracRaster, R_frac] = readgeoraster(fileRasterFrac);
        georefRaster = readgeoraster(fileGeoRef);            % polygon index of every pixel
        peatmap = shaperead(filePeatShp);

        fracRaster = single(fracRaster);

        for j = 1:length(peatmap)
            thisID = peatmap(j).ID;
            ind = find(georefRaster == thisID);
            if isempty(ind)
                continue;
            end

            permBox = permWater(ind);
            permPer = sum(permBox == 1) / numel(ind) * 100;     % % of permanent water in the polygon

            % a polygon that is all water gets 100 % but has no land pixel, so no peat is placed in it
            newFrac = min(peatmap(j).PEAT_PER / (100 - permPer) * 100, 100);
            fracRaster(ind) = newFrac;
            peatmap(j).PEAT_PER = newFrac;
        end

        geotiffwrite(fileRasterFracC, fracRaster, R_frac);
        fprintf('SAVED %s\n', fileRasterFracC);

        shapewrite(peatmap, filePeatShpC);
        fprintf('SAVED %s\n', filePeatShpC);
    else
        if ~hasWater
            fprintf('Skip peat fraction correction for %s (no water data).\n', regionName);
        else
            fprintf('Skip peat fraction correction for %s (missing frac/shp).\n', regionName);
        end
    end

    %% MERIT inputs on the tile grid
    meritLayers = {'dist0005', 'dest_0005'; 'dist0100', 'dest_0100'; 'dist1000', 'dest_1000'; ...
                   'hand0005', 'hand_0005'; 'hand0100', 'hand_0100'; 'hand1000', 'hand_1000'; ...
                   'slope', 'slope'; 'elevation', 'elevation'};
    for k = 1:size(meritLayers, 1)
        name = meritLayers{k, 1};
        myFile = fullfile(processedDataFolder, [name '.mat']);
        fprintf('Processing %s for %s\n', name, regionName);
        try
            [v, latD, lonD] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, meritLayers{k, 2});
            S = struct();
            S.(name) = interp2(lonD, latD, v, newLon, newLat);
            S.R = R;
            save(myFile, '-struct', 'S', '-v7.3');
            fprintf('SAVED %s\n', myFile);
        catch ME
            fprintf('Failed to process %s for %s: %s\n', name, regionName, ME.message);
        end
    end

    fprintf('Completed processing for %s\n', regionName);
end

fprintf('\nALL DONE\n');

%% Canada: rasterise the Peatlands of Canada polygons (PEATMAP) on the Landsat tile grid
% Input:  Raw/2_PEATMAP/Canada_Projected.shp (polygons with PEAT_PER = % peat and AREA)
%         the Landsat 8 tiles of North America (1_inputs/LS8_DP_multiple.m), which define the 90 m tile grids
% Output: Processed/2_PEATMAP/Canada/Tiles/<tile>/
%         <tile>_georef.tif  polygon index of every pixel (0 = no polygon)
%         <tile>_frac.tif    peat fraction (%) of the polygon
%         <tile>_area.tif    area of the polygon
%         <tile>_latlon.shp  polygons that touch the tile
% A pixel belongs to a polygon when its centre is inside it (inpoly2). Requires inpoly2 (D. Engwirda) on the MATLAB path.
% Next: 1_inputs/MERIT_DP_Canada.m (permanent water, water-corrected fractions), then BuildTrainingData_Canada.m.

clear
clc

root = getenv('PEATCOVER_ROOT');
filePeatmap      = fullfile(root, 'Raw/2_PEATMAP/Canada_Projected.shp');
landsatFolder    = fullfile(root, 'Processed/1_Landsat8/Landsat8_NA_July/');
outputBaseFolder = fullfile(root, 'Processed/2_PEATMAP/Canada/Tiles');

fprintf('LOADING %s\n', filePeatmap);
peatmap = shaperead(filePeatmap);

% bounding box and index of every polygon
toto = num2cell(arrayfun(@(x) min(x.Y), peatmap)); [peatmap(:).minLat] = deal(toto{:});
toto = num2cell(arrayfun(@(x) max(x.Y), peatmap)); [peatmap(:).maxLat] = deal(toto{:});
toto = num2cell(arrayfun(@(x) min(x.X), peatmap)); [peatmap(:).minLon] = deal(toto{:});
toto = num2cell(arrayfun(@(x) max(x.X), peatmap)); [peatmap(:).maxLon] = deal(toto{:});
toto = num2cell((1:length(peatmap))');             [peatmap(:).ID] = deal(toto{:});

landsatFiles = dir(fullfile(landsatFolder, '*.mat'));
for fileIdx = 1:length(landsatFiles)
    landsatFilePath = fullfile(landsatFiles(fileIdx).folder, landsatFiles(fileIdx).name);
    fprintf('PROCESSING %s\n', landsatFiles(fileIdx).name);

    load(landsatFilePath, 'R', 'info');
    res = R.CellExtentInLatitude;

    maxLat = max(info.CornerCoords.Lat);
    minLat = min(info.CornerCoords.Lat);
    maxLon = max(info.CornerCoords.Lon);
    minLon = min(info.CornerCoords.Lon);
    if maxLon > 180, maxLon = maxLon - 360; end
    if minLon > 180, minLon = minLon - 360; end

    % polygons that touch the tile
    if minLon > maxLon      % tile crosses the 180 deg meridian
        regionPeatmap = peatmap(([peatmap.minLat] < maxLat & [peatmap.maxLat] > minLat) & ...
                                (([peatmap.minLon] < maxLon) | ([peatmap.maxLon] > minLon)));
    else
        regionPeatmap = peatmap([peatmap.minLat] < maxLat & [peatmap.maxLat] > minLat & ...
                                [peatmap.minLon] < maxLon & [peatmap.maxLon] > minLon);
    end
    fprintf('Number of PEATMAP polygons in the tile: %d\n', length(regionPeatmap));
    if isempty(regionPeatmap)
        continue;
    end

    [~, regionName, ~] = fileparts(landsatFiles(fileIdx).name);
    outputFolder = fullfile(outputBaseFolder, regionName);
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    filePeatmapPoly2 = fullfile(outputFolder, [regionName '_latlon.shp']);
    fileRasterGeoref = fullfile(outputFolder, [regionName '_georef.tif']);
    fileRasterFrac   = fullfile(outputFolder, [regionName '_frac.tif']);
    fileRasterArea   = fullfile(outputFolder, [regionName '_area.tif']);

    nbRow = R.RasterSize(1);
    nbCol = R.RasterSize(2);
    georefRaster = zeros(nbRow, nbCol, 'uint32');
    fracRaster   = zeros(nbRow, nbCol, 'uint16');
    areaRaster   = zeros(nbRow, nbCol, 'single');

    % pixel centres
    lat = linspace(R.LatitudeLimits(2) - R.CellExtentInLatitude / 2, ...
                   R.LatitudeLimits(1) + R.CellExtentInLatitude / 2, nbRow);
    lon = linspace(R.LongitudeLimits(1) + R.CellExtentInLongitude / 2, ...
                   R.LongitudeLimits(2) - R.CellExtentInLongitude / 2, nbCol);
    [lon, lat] = meshgrid(lon, lat);

    %% Rasterise in blocks of 500 x 500 pixels
    blockSize = 500;
    listRowStart = 1:blockSize:nbRow;
    listColStart = 1:blockSize:nbCol;
    for iR = 1:length(listRowStart)
        for iC = 1:length(listColStart)
            rowStart = listRowStart(iR);
            rowStop  = min(rowStart + blockSize - 1, nbRow);
            colStart = listColStart(iC);
            colStop  = min(colStart + blockSize - 1, nbCol);

            blockLat = lat(rowStart:rowStop, colStart:colStop);
            blockLon = lon(rowStart:rowStop, colStart:colStop);

            minLat = min(blockLat(:)) - res/2;
            maxLat = max(blockLat(:)) + res/2;
            minLon = min(blockLon(:)) - res/2;
            maxLon = max(blockLon(:)) + res/2;
            localShape = peatmap([peatmap.minLat] < maxLat & [peatmap.maxLat] > minLat & ...
                                 [peatmap.minLon] < maxLon & [peatmap.maxLon] > minLon);
            if isempty(localShape)
                continue;
            end

            georefBlock = zeros(size(blockLat), 'uint32');
            fracBlock   = zeros(size(blockLat), 'uint8');
            areaBlock   = zeros(size(blockLat), 'single');

            % polygons fully inside the block
            inside = ([localShape.minLat] >= minLat & [localShape.maxLat] <= maxLat & ...
                      [localShape.minLon] >= minLon & [localShape.maxLon] <= maxLon);
            for iPoly = find(inside == 1)
                [pixInside, ~] = inpoly2([blockLon(:), blockLat(:)], ...
                                         [localShape(iPoly).X(:), localShape(iPoly).Y(:)]);
                georefBlock(pixInside) = localShape(iPoly).ID;
                fracBlock(pixInside)   = localShape(iPoly).PEAT_PER;
                areaBlock(pixInside)   = localShape(iPoly).AREA;
            end

            % polygons that cross the block edge: clip to the block first; if clipping fails, use the whole polygon
            for iPoly = find(inside == 0)
                try
                    [trim_Y, trim_X] = maptrimp(localShape(iPoly).Y, localShape(iPoly).X, ...
                                                double([minLat maxLat]), double([minLon maxLon]));
                    if isempty(trim_X)
                        continue;
                    end
                    [pixInside, ~] = inpoly2([blockLon(:), blockLat(:)], [trim_X(:), trim_Y(:)]);
                    georefBlock(pixInside) = localShape(iPoly).ID;
                    fracBlock(pixInside)   = localShape(iPoly).PEAT_PER;
                    areaBlock(pixInside)   = localShape(iPoly).AREA;
                catch ME
                    fprintf('Polygon %d: maptrimp failed (%s), using the unclipped polygon\n', ...
                            localShape(iPoly).ID, ME.message);
                    try
                        X = localShape(iPoly).X;
                        Y = localShape(iPoly).Y;
                        valid = ~isnan(X) & ~isnan(Y);
                        X = X(valid);
                        Y = Y(valid);
                        if ~isempty(X) && max(Y) >= minLat && min(Y) <= maxLat && ...
                                          max(X) >= minLon && min(X) <= maxLon
                            [pixInside, ~] = inpoly2([blockLon(:), blockLat(:)], [X(:), Y(:)]);
                            georefBlock(pixInside) = localShape(iPoly).ID;
                            fracBlock(pixInside)   = localShape(iPoly).PEAT_PER;
                            areaBlock(pixInside)   = localShape(iPoly).AREA;
                        end
                    catch ME2
                        fprintf('Polygon %d skipped: %s\n', localShape(iPoly).ID, ME2.message);
                        continue;
                    end
                end
            end

            georefRaster(rowStart:rowStop, colStart:colStop) = georefBlock;
            fracRaster(rowStart:rowStop, colStart:colStop)   = fracBlock;
            areaRaster(rowStart:rowStop, colStart:colStop)   = areaBlock;
        end
    end

    %% Save
    shapewrite(regionPeatmap, filePeatmapPoly2);
    geotiffwrite(fileRasterGeoref, georefRaster, R);
    geotiffwrite(fileRasterFrac, fracRaster, R);
    geotiffwrite(fileRasterArea, areaRaster, R);
    fprintf('SAVED %s\n', outputFolder);
    clear georefRaster fracRaster areaRaster lat lon
end
fprintf('ALL DONE\n');

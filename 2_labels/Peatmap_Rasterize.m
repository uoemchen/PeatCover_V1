%% Rasterise the PEATMAP tile polygons (cut_peatmap_tiles.py) on the Landsat tile grid: 1 = peat, 0 = other
% A pixel is peat when its centre is inside a polygon (inpoly2, D. Engwirda, must be on the MATLAB path).
% Output: <output_folder>/<tile>/<tile>_PEATMASK.tif. Set region to 'europe' or 'usa'.

clear
clc

region = 'europe';
root = getenv('PEATCOVER_ROOT');
if strcmp(region, 'europe')
    landsat_folder = fullfile(root, 'Raw/1_Landsat8/Landsat8_Europe_July');
    peatmap_folder = fullfile(root, 'Processed/2_PEATMAP/Europe/Tile_Shapefiles');
    output_folder  = fullfile(root, 'Processed/2_PEATMAP/Europe/Tile_Rasters');
else
    landsat_folder = fullfile(root, 'Raw/1_Landsat8/Landsat8_NA_July');
    peatmap_folder = fullfile(root, 'Processed/2_PEATMAP/USA/Tile_Shapefiles');
    output_folder  = fullfile(root, 'Processed/2_PEATMAP/USA/Tile_Rasters');
end
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

landsat_files = dir(fullfile(landsat_folder, '*.tif'));
for fileIdx = 1:length(landsat_files)
    landsat_file = fullfile(landsat_files(fileIdx).folder, landsat_files(fileIdx).name);
    [~, R] = readgeoraster(landsat_file);
    res = R.CellExtentInLatitude;

    tile_name = split(landsat_files(fileIdx).name, '_');
    tile_name = erase(tile_name{end}, '.tif');                 % e.g. N60E025
    peatmap_file = fullfile(peatmap_folder, tile_name, sprintf('%s_PEATMAP.shp', tile_name));
    if ~isfile(peatmap_file)
        continue;
    end
    fprintf('PROCESSING %s\n', tile_name);
    S = shaperead(peatmap_file);

    nbRow = R.RasterSize(1);
    nbCol = R.RasterSize(2);
    fracRaster = zeros(nbRow, nbCol, 'int8');

    % pixel centres
    lat = linspace(R.LatitudeLimits(2) - res/2, R.LatitudeLimits(1) + res/2, nbRow);
    lon = linspace(R.LongitudeLimits(1) + res/2, R.LongitudeLimits(2) - res/2, nbCol);
    [lon, lat] = meshgrid(lon, lat);

    % blocks of 500 x 500 pixels
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
            fracBlock = zeros(size(blockLat), 'int8');
            for iPoly = 1:length(S)
                [pixInside, ~] = inpoly2([blockLon(:), blockLat(:)], [S(iPoly).X(:), S(iPoly).Y(:)]);
                fracBlock(pixInside) = 1;
            end
            fracRaster(rowStart:rowStop, colStart:colStop) = fracBlock;
        end
    end

    outputTileFolder = fullfile(output_folder, tile_name);
    if ~exist(outputTileFolder, 'dir')
        mkdir(outputTileFolder);
    end
    outputRasterFile = fullfile(outputTileFolder, sprintf('%s_PEATMASK.tif', tile_name));
    geotiffwrite(outputRasterFile, fracRaster, R);
    fprintf('SAVED %s\n', outputRasterFile);
end
fprintf('ALL DONE\n');

%% Landsat 8 preprocessing
% Converts the Landsat 8 tiles exported from Google Earth Engine (one GeoTIFF per 5 x 5 deg tile with a 20-pixel buffer,
% 7 surface reflectance bands of the July composite, file name Landsat8_<start year>_<end year>_<month>_<tile>.tif)
% into one .mat file per tile: allL8 (pixels x 7), R (raster reference) and info.
% Regions: Europe, NA and Dateline (the tiles across 180 deg, exported separately).

root = getenv('PEATCOVER_ROOT');
regions = {'Europe', 'NA', 'Dateline'};

for r = 1:numel(regions)
    inputFolder = fullfile(root, 'Raw/1_Landsat8', ['Landsat8_' regions{r} '_July']);
    outputFolder = fullfile(root, 'Processed/1_Landsat8', ['Landsat8_' regions{r} '_July']);
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    process_LS8_multiple(inputFolder, outputFolder);
end

function process_LS8_multiple(folderPath, outputFolder)
    % Processes every Landsat 8 tile of folderPath as a batch job, with at most maxConcurrentJobs jobs at a time.
    files = dir(fullfile(folderPath, 'Landsat8_*.tif'));
    numFiles = length(files);

    maxConcurrentJobs = 2;
    runningJobs = [];

    for i = 1:numFiles
        filePath = fullfile(files(i).folder, files(i).name);
        job = batch(@process_single_LS8, 0, {filePath, outputFolder});
        runningJobs = [runningJobs; job];

        % wait until a job has finished before submitting the next one
        while length(runningJobs) >= maxConcurrentJobs
            pause(2);
            isFinished = arrayfun(@(j) strcmp(j.State, 'finished'), runningJobs);
            runningJobs(isFinished) = [];
        end
    end

    for i = 1:length(runningJobs)
        wait(runningJobs(i));
    end

    fprintf('All processing completed. Total files processed: %d\n', numFiles);
end


function process_single_LS8(filePath, outputFolder)
    % One Landsat 8 tile -> <tile>.mat
    filename = split(filePath, '/');
    filename = filename{end};
    parts = split(filename, '_');
    year = parts{2};                          % start year of the composite
    end_year = parts{3};                      % end year
    month = parts{4};                         % month
    regionID = erase(parts{5}, '.tif');       % tile name, e.g. N60E025

    outputFileName = sprintf('%s.mat', regionID);
    outputFilePath = fullfile(outputFolder, outputFileName);

    [L8Data, R] = readgeoraster(filePath);

    if size(L8Data, 3) < 7
        error('Image is missing bands, please check the data source.');
    end

    allL8 = reshape(L8Data, [], 7);

    info.CornerCoords.Lat = R.LatitudeLimits;
    info.CornerCoords.Lon = R.LongitudeLimits;
    info.PixelScale = [R.CellExtentInLatitude, R.CellExtentInLongitude];
    info.Height = R.RasterSize(1);
    info.Width = R.RasterSize(2);

    save(outputFilePath, 'allL8', 'R', 'info', '-v7.3');
    fprintf('Saved %s\n', outputFilePath);
end

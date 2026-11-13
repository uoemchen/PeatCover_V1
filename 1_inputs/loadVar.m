function variable = loadVar(folderPath, regionName, varName)
% Load one input variable of a tile.
%   loadVar(folder, tile, name)   reads <folder>/<tile>/<name>.mat and returns the variable <name>
%   loadVar(file, '', 'landsat')  reads a Landsat tile file and returns allL8
    if isempty(regionName)
        inFile = folderPath;
    else
        inFile = fullfile(folderPath, regionName, [varName '.mat']);
    end

    fprintf('LOADING %s\n', inFile);

    if ~isfile(inFile)
        error('File not found: %s', inFile);
    end

    data = load(inFile);

    if strcmp(varName, 'landsat') && isfield(data, 'allL8')
        variable = data.allL8;
    elseif isfield(data, varName)
        variable = data.(varName);
    else
        error('Variable ''%s'' not found in file: %s', varName, inFile);
    end
end

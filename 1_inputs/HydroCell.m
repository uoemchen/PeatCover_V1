classdef HydroCell
% One 5 x 5 deg MERIT tile (3 arc-seconds, 6000 x 6000 pixels).
% Layers (ext) and files below folderPath:
%   'elevation'             Yamazaki/MERIT/v0.4_original_data_distributed/elevtn/<cell>_elv.tif   (MERIT DEM)
%   'wth'                   Yamazaki/MERIT/v0.4_original_data_distributed/width/<cell>_wth.tif    (MERIT Hydro river width)
%   'slope'                 Yamazaki/MERIT/v0.4_diagnostics_old_tif/slope_tif/<cell>.tif
%   'dest_*', 'hand_*'      Yamazaki2/<ext>/<cell>.tif   (distance to / height above the drainage network)
% <cell> is the name of the tile, e.g. n60e025.

properties
    x;              % longitude of the pixel centres
    y;              % latitude of the pixel centres
    data;
    nrows=6000;
    ncols=6000;
    nbits=32;
    dtype;
    nodata=-9999;
    ulxmap;         % longitude at the centre of the upper-left pixel
    ulymap;         % latitude at the centre of the upper-left pixel
    xdim=3/3600;
    ydim=3/3600;
    cellLat;        % -90 to 90
    cellLon;        % -180 to 180
    folderPath;
    cellName;
    fileType;
end

methods
    function obj=HydroCell(folderPath, cellLat,cellLon,ext)

        obj.folderPath = folderPath;

        obj.cellLat=cellLat;
        if (cellLat>90)
           obj.cellLat=cellLat-90;
        end
        obj.cellLon=cellLon;
        if (cellLon>=180)
            obj.cellLon=cellLon-360;
        elseif (cellLon<-180)
            obj.cellLon=cellLon+360;
        end

        latLetter='n';
        if (obj.cellLat<0)
            latLetter='s';
        end
        lonLetter='e';
        if (obj.cellLon<0)
            lonLetter='w';
        end

        obj.cellName = strcat(latLetter,sprintf('%02d',abs(obj.cellLat)),lonLetter,sprintf('%03d',abs(obj.cellLon)));
        obj.fileType = ext;

        obj.ulxmap=obj.cellLon;
        obj.ulymap=obj.cellLat+5-obj.ydim;     % MERIT grid
        obj.dtype = 'float';

        if strcmp(ext, 'slope')
            ftif = fullfile(obj.folderPath, 'Yamazaki/MERIT/v0.4_diagnostics_old_tif/slope_tif', [obj.cellName '.tif']);
        elseif strcmp(ext, 'wth')
            ftif = fullfile(obj.folderPath, 'Yamazaki/MERIT/v0.4_original_data_distributed/width', [obj.cellName '_wth.tif']);
        elseif strcmp(ext, 'elevation')
            ftif = fullfile(obj.folderPath, 'Yamazaki/MERIT/v0.4_original_data_distributed/elevtn', [obj.cellName '_elv.tif']);
        elseif (strcmp(ext(1:3), 'des') || strcmp(ext(1:3), 'han'))
            ftif = fullfile(obj.folderPath, 'Yamazaki2', ext, [obj.cellName '.tif']);
        else
            error('Unknown MERIT layer: %s', ext);
        end

        if exist(ftif, 'file')
            [obj.data,~] = geotiffread(ftif);
            tifInfo      = geotiffinfo(ftif);
            obj.nrows    = tifInfo.Height;
            obj.ncols    = tifInfo.Width;
            obj.nbits    = tifInfo.BitDepth;
            obj.xdim     = tifInfo.PixelScale(1);
            obj.ydim     = tifInfo.PixelScale(2);
            obj.ulxmap   = tifInfo.BoundingBox(1,1)+obj.xdim/2;
            obj.ulymap   = tifInfo.BoundingBox(2,2)-obj.ydim/2;
            obj.dtype    = sprintf('float%u',obj.nbits);
            obj.nodata   = -9999;
            obj.data(obj.data==obj.nodata) = NaN;
        else
            fprintf('File "%s" does not exist. \n',ftif);
            obj.data=nan(obj.nrows,obj.ncols);
        end

        % coordinates of the pixel centres
        obj.x = obj.ulxmap+obj.xdim*(0:obj.ncols-1);
        obj.y = obj.ulymap-obj.ydim*(0:obj.nrows-1)';
    end
end

end

function [windo, lat, lon] = windowCell(path, latMin0, latMax0, lonMin0, lonMax0, ext)
% Returns the MERIT layer 'ext' over the window [latMin0, latMax0] x [lonMin0, lonMax0], assembled from the 5 x 5 deg
% MERIT tiles (HydroCell.m), with the latitude and longitude of the pixel centres.

dsharp = 3/3600;                              % 3 arc-seconds in degrees
cellSize = 5/dsharp;

latMin    = floor(latMin0/dsharp+0.5)*dsharp;
latMax    = floor(latMax0/dsharp+0.5)*dsharp;
lonMin    = floor(lonMin0/dsharp+0.5)*dsharp;
lonMax    = floor(lonMax0/dsharp+0.5)*dsharp;

nRows     = floor((latMax-latMin)/dsharp+0.5);
nCols     = floor((lonMax-lonMin)/dsharp+0.5);
windo     = zeros(nRows,nCols, 'single');

latBoxMax = ceil (latMax/5)*5;
latBoxMin = floor(latMin/5)*5;
lonBoxMax = ceil (lonMax/5)*5;
lonBoxMin = floor(lonMin/5)*5;

for lat=latBoxMin:5:(latBoxMax-5)
    for lon=lonBoxMin:5:(lonBoxMax-5)
        cell=HydroCell(path, lat,lon,ext);
        padding_left   = floor((lonMin-lon)/dsharp+0.5);
        padding_right  = floor((lonMax-lon)/dsharp+0.5);
        padding_top    = -floor((latMax-lat-5)/dsharp+0.5);
        padding_bottom = -floor((latMin-lat-5)/dsharp+0.5);
        windo(max(1,1-padding_top):min(nRows,floor((latMax-lat)/dsharp+0.5)), ...
              max(1,1-padding_left):min(nCols,floor((lon+5-lonMin)/dsharp+0.5))) ...
      = cell.data(max(1,1+padding_top):min(cellSize,padding_bottom), ...
                  max(1,padding_left+1):min(cellSize,padding_right));
    end
end

lat = single(latMax + dsharp/2 - dsharp*(1:nRows));   % pixel centres
lon = single(lonMin - dsharp/2 + dsharp*(1:nCols));

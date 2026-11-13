function [wtd, lat, lon, mask] = waterTableDepth(WTD_folder, minLat, maxLat, minLon, maxLon, myZone, plottt)
% Annual mean water table depth (m, positive downwards) of Fan et al. (2013) over the window
% [minLat, maxLat] x [minLon, maxLon]; myZone = 'EURASIA' or 'NAMERICA' (file <myZone>_WTD_annualmean.nc).
% Returns the WTD (NaN outside the land mask), the latitudes (north to south), longitudes and the land mask.
% plottt is not used.

WTD_file = fullfile(WTD_folder, [char(myZone) '_WTD_annualmean.nc']);

lon = ncread(WTD_file, 'lon');
lat = ncread(WTD_file, 'lat');           % south to north
[~, indLatStart] = min(abs(lat-minLat));
[~, indLatStop]  = min(abs(lat-maxLat));
[~, indLonStart] = min(abs(lon-minLon));
[~, indLonStop]  = min(abs(lon-maxLon));

lat = single(lat(indLatStart:indLatStop)); lat = flipud(lat);
lon = single(lon(indLonStart:indLonStop));

mask = ncread(WTD_file, 'mask', [indLonStart indLatStart], [length(lon) length(lat)]);
mask = rot90(mask);

wtd = ncread(WTD_file, 'WTD', [indLonStart indLatStart 1], [length(lon) length(lat) Inf]);
wtd = single(rot90(wtd)*-1);
wtd(~mask) = NaN;

end

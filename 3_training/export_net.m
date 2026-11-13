function export_net(net, outFile)
% Write the weights and input scaling of a trained patternnet to a plain .mat file, so that the same network can be
% applied in Python (4_mapping/matlab_net.py) exactly as MATLAB does:
%   x -> removeconstantrows -> mapminmax ((x - xoffset) .* gain + ymin) -> tansig hidden layer(s) -> softmax output
% Output row 2 of the softmax is the peat probability (= Peatland Index / 100).
% Regression network (one output row): the output scaling is exported too (out_*).
nL = net.numLayers;
W = cell(1, nL); b = cell(1, nL); tf = cell(1, nL);
W{1} = net.IW{1, 1};
for l = 2:nL
    W{l} = net.LW{l, l - 1};
end
for l = 1:nL
    b{l} = net.b{l};
    tf{l} = net.layers{l}.transferFcn;
end
pf = net.inputs{1}.processFcns;
ps = net.inputs{1}.processSettings;
keep = 1:net.inputs{1}.size;
xoffset = []; gain = []; ymin = [];
for k = 1:numel(pf)
    switch pf{k}
        case 'removeconstantrows'
            keep = ps{k}.keep;
        case 'mapminmax'
            xoffset = ps{k}.xoffset; gain = ps{k}.gain; ymin = ps{k}.ymin;
    end
end
outFcns = net.outputs{nL}.processFcns;   % patternnet: removeconstantrows only, identity for 2 classes
% one target row (regression, 05_canada_regression_vb.py): mapminmax of the target, reversed on the output: x = (y - ymin) / gain + xoffset
out_xoffset = []; out_gain = []; out_ymin = [];
for k = 1:numel(outFcns)
    if strcmp(outFcns{k}, 'mapminmax')
        s = net.outputs{nL}.processSettings{k};
        out_xoffset = s.xoffset; out_gain = s.gain; out_ymin = s.ymin;
    end
end
save(outFile, 'W', 'b', 'tf', 'keep', 'xoffset', 'gain', 'ymin', 'outFcns', 'out_xoffset', 'out_gain', 'out_ymin', '-v7');
end

function trainLattice(path)

    latticeElements = dirPattern([path '/*.json']);

    out = struct([]);
    for i = 1:numel(latticeElements)
        out(i).settings = loadjson([path '/' latticeElements{i}]);
        [out(i).accuracy, out(i).results, out(i).models] = trainSVM(out(i).settings);
        save([path '/lattice.mat'], 'out');
    end

end
latticeElements = dirPattern('./lattice/*.json');

for i = 1:numel(latticeElements)
    out(i).settings = loadjson(latticeElements{i});
    [out(i).accuracy, out(i).results, out(i).models] = trainSVM(out(i).settings);
end

save('lattice.mat', 'out');

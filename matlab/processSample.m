function data_bands = processSample(filename, settings)
    % Data dimensions: (channel, time, window)

    data = load(filename);
    name = fieldnames(data);
    data = data.(name{1});

    % Windowing

    fs = data.sampling_frequency;
    [nch, nsamp] = size(data.data);
    winlen = round(fs * settings.window.length);
    if mod(winlen,2)
        winlen = winlen + 1;
    end
    nwindows = floor(nsamp / winlen);
    overlap = round(fs * settings.window.overlap);

    raw = zeros(nch, winlen, nwindows);
    for i = 1:nwindows
        raw(:,:,i) = data.data(:, overlap*(i-1)+1 : overlap*(i-1)+winlen);
    end

    % Normalize frequency energies
    normalized = diff(raw, 1, 2);

    % FFT mangnitude and band analysis
    nbins = winlen/2;
    freq = abs(fft(normalized, [], 2));
    freq = freq(:,1:nbins,:);

    nbands = settings.bands.count;
    bandw = floor(nbins / nbands);
    data_bands = zeros(nch, nbands, nwindows);
    for i = 1:nbands
        data_bands(:,i,:) = sum(freq(:,(i-1)*bandw+1:i*bandw+1, :), 2);
    end

end
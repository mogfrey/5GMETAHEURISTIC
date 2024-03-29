function whalePopulation = initializewhale(numWhales)
    % Initialize whale positions within specified ranges
    whalePopulation = zeros(numWhales, 4); % 4 parameter to optimize
    for i = 1:numWhales
        whalePopulation(i, 1) = randi([50, 106]); % NSizeGrid
        whalePopulation(i, 2) = randsample([15, 30, 60, 120], 1); % SubcarrierSpacing
        %whalePopulation(i, 3) = (500e-9 - 50e-9) * rand() + 50e-9; % DelaySpread (50 ns to 500 ns)
        whalePopulation(i, 4) = 30 * rand(); % MaximumDopplerShift (0 to 30 Hz)
    end
end
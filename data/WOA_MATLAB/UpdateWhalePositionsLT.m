function updatedWhalePopulation = UpdateWhalePositionsLT(whalePopulation, whaleProcessingTimes, paramRanges, a, a2)
    % Update Whale Positions based on processing time

    % Parameters:
    % whalePopulation: Current positions of whales
    % whaleProcessingTimes: Processing time of each whale
    % paramRanges: Ranges of parameters being optimized
    % a: Coefficient that decreases linearly from 2 to 0 over iterations
    % a2: Random number in [-1, 1], for random search

    validSubcarrierSpacings = [15, 30, 60, 120, 240]; % Valid SubcarrierSpacing values

    % Find the best whale (with minimum processing time)
    [bestProcessingTime, bestIdx] = min(whaleProcessingTimes);
    bestWhale = whalePopulation(bestIdx, :);

    numWhales = size(whalePopulation, 1);
    numParams = size(whalePopulation, 2); % Adjusted for the new parameters
    updatedWhalePopulation = whalePopulation;

    for i = 1:numWhales
        for j = 1:numParams
            r = rand(); % Random number in [0,1]
            A = 2 * a * r - a; % A decreases from 2 to 0 over iterations
            C = 2 * rand();
            D = abs(C * bestWhale(j) - whalePopulation(i, j));

            if j == 2 % Special handling for SubcarrierSpacing
                updatedPosition = whalePopulation(i, j) - A * D;
                % Find the nearest valid value from the set
                [~, nearestIdx] = min(abs(validSubcarrierSpacings - updatedPosition));
                updatedWhalePopulation(i, j) = validSubcarrierSpacings(nearestIdx);
            elseif j == 3 % Special handling for DelaySpread
                updatedWhalePopulation(i, j) = whalePopulation(i, j) - A * D;
                updatedWhalePopulation(i, j) = max(paramRanges{j}(1), updatedWhalePopulation(i, j));
                updatedWhalePopulation(i, j) = min(paramRanges{j}(2), updatedWhalePopulation(i, j));
            elseif j == 4 % Special handling for MaximumDopplerShift
                updatedWhalePopulation(i, j) = whalePopulation(i, j) - A * D;
                updatedWhalePopulation(i, j) = max(paramRanges{j}(1), updatedWhalePopulation(i, j));
                updatedWhalePopulation(i, j) = min(paramRanges{j}(2), updatedWhalePopulation(i, j));
            else % Standard updating logic for NSizeGrid
                if rand() < 0.5
                    updatedWhalePopulation(i, j) = bestWhale(j) - A * D;
                else
                    b = 1; % Defines shape of the spiral
                    l = (a2 - 1) * rand() + 1;
                    updatedWhalePopulation(i, j) = D * exp(b * l) * cos(2 * pi * l) + bestWhale(j);
                end
                updatedWhalePopulation(i, j) = max(paramRanges{j}(1), updatedWhalePopulation(i, j));
                updatedWhalePopulation(i, j) = min(paramRanges{j}(2), updatedWhalePopulation(i, j));
            end
        end
    end
end

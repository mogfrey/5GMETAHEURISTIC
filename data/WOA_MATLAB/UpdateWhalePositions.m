function updatedWhalePopulation = UpdateWhalePositions(whalePopulation, whaleThroughputs, paramRanges, a, a2)
    % Update Whale Positions based on WOA

    % Parameters:
    % whalePopulation: Current positions of whales
    % whaleThroughputs: Throughput of each whale
    % paramRanges: Ranges of parameters being optimized
    % a: Coefficient that decreases linearly from 2 to 0 over iterations
    % a2: Random number in [-1, 1], for random search

    % Find the best whale
    [bestThroughput, bestIdx] = max(whaleThroughputs);
    bestWhale = whalePopulation(bestIdx, :);

    numWhales = size(whalePopulation, 1);
    numParams = size(whalePopulation, 2);
    updatedWhalePopulation = whalePopulation;

    for i = 1:numWhales
        r = rand();  % Random number in [0,1]
        A = 2 * a * r - a;  % A decreases from 2 to 0 over iterations
        C = 2 * rand();
        b = 1;  % Defines shape of the spiral
        l = (a2 - 1) * rand() + 1;

        for j = 1:numParams
            D = abs(C * bestWhale(j) - whalePopulation(i, j));

            if rand() < 0.5
                if abs(A) < 1
                    % Shrinking encircling mechanism
                    updatedWhalePopulation(i, j) = bestWhale(j) - A * D;
                else
                    % Exploratory search mechanism
                    randomWhale = whalePopulation(randi(numWhales), j);
                    updatedWhalePopulation(i, j) = randomWhale - A * D;
                end
            else
                % Spiral updating position
                updatedWhalePopulation(i, j) = D * exp(b * l) * cos(2 * pi * l) + bestWhale(j);
            end

            % Ensure the new position is within bounds
            updatedWhalePopulation(i, j) = max(paramRanges{j}(1), updatedWhalePopulation(i, j));
            updatedWhalePopulation(i, j) = min(paramRanges{j}(2), updatedWhalePopulation(i, j));

            % Special handling for SubcarrierSpacing
            if j == 2
                validSubcarrierSpacings = [15, 30, 60, 120, 240];
                [~, nearestIdx] = min(abs(validSubcarrierSpacings - updatedWhalePopulation(i, j)));
                updatedWhalePopulation(i, j) = validSubcarrierSpacings(nearestIdx);
            end
        end
    end
end

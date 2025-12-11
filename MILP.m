clc;clear

%% 1. Build MILP model
% Define decision variables
C_pv = intvar(1,1);             % PV capacity (integer)
C_wt = intvar(1,1);             % Wind turbine capacity (integer)
E_bat = sdpvar(1,1);            % Battery capacity
P_bat = sdpvar(1,1);            % Battery charge/discharge power

% Define hourly operation variables
p_pv = sdpvar(8760,1);          % Actual PV output
p_wt = sdpvar(8760,1);          % Actual wind turbine output
p_bat_ch = sdpvar(8760,1);      % Battery charging power
p_bat_dis = sdpvar(8760,1);     % Battery discharging power
p_buy = sdpvar(8760,1);         % Grid purchase power
SOC = sdpvar(8760,1);           % Battery state of charge (SOC)
mg_load = sdpvar(8760,1);       % Microgrid local load
price_mg = 0.33;                % Industrial park electricity price

%% 2. Generate PV output data
% Read Excel data
pv_data = readmatrix('cluster pv.xlsx');

% Extract column vectors
clu_pv1 = pv_data(:, 1);
clu_pv2 = pv_data(:, 2);
clu_pv3 = pv_data(:, 3);
vectors = {clu_pv1, clu_pv2, clu_pv3}; % Store in cell array for easy indexing

% Probability distributions for each season
p_spring = [15.21, 50.19, 34.6] / 100;
p_summer = [61.96, 17.44, 20.6] / 100;
p_autumn = [2.2, 85.73, 12.07] / 100;
p_winter = [0, 82.28, 17.72] / 100;

% Generator: create vector blocks according to probability distribution
generatePart = @(prob, n) ...
    reshape([vectors{randsample(1:3, n, true, prob)}], [], 1);

% Generate vectors for each season and concatenate
part_spring = generatePart(p_spring, 92);
part_summer = generatePart(p_summer, 92);
part_autumn = generatePart(p_autumn, 91);
part_winter = generatePart(p_winter, 90);

% Concatenate all parts to form the final vector
predict_pv = [part_spring; part_summer; part_autumn; part_winter];

% Add random fluctuations
noise_intensity = 0.05; % Noise intensity coefficient
predict_pv = predict_pv .* (1 + noise_intensity*randn(size(predict_pv)));

% Data correction
predict_pv(predict_pv < 0) = 0; % Remove negative values
predict_pv = min(predict_pv, 1.2*max(pv_data(:))); % Limit maximum output

%% 3. Generate wind turbine output data
% Read Excel data
wt_data = readmatrix('cluster wind.xlsx');

% Extract column vectors
clu_wt1 = wt_data(:, 1);
clu_wt2 = wt_data(:, 2);
clu_wt3 = wt_data(:, 3);
clu_wt4 = wt_data(:, 4);
vectors_wt = {clu_wt1, clu_wt2, clu_wt3, clu_wt4}; % Store in cell array for easy indexing

% Probability distributions for each season
p_spr = [26.64, 24.19, 13.23, 35.94] / 100;
p_sum = [38.02, 16.88, 4.06, 41.04] / 100;
p_aut = [42.49, 23, 15.28, 19.23] / 100;
p_win = [29.9, 29.83, 15.6, 24.67] / 100;

% Generator: create vector blocks according to probability distribution
generatePart_wt = @(prob, n) ...
    reshape([vectors_wt{randsample(1:4, n, true, prob)}], [], 1);

% Generate vectors for each season and concatenate
part_spr = generatePart_wt(p_spr, 92);
part_sum = generatePart_wt(p_sum, 92);
part_aut = generatePart_wt(p_aut, 91);
part_win = generatePart_wt(p_win, 90);

% Concatenate all parts to form the final vector
predict_wt = [part_spr; part_sum; part_aut; part_win];

% Generate operating output according to wind turbine power curve
actual_wt = zeros(size(predict_wt));   % Initialize output to 0 for wind speed below 2.2 m/s

idx1 = predict_wt >= 2.2;              % Process wind speed above 2.2 m/s
actual_wt(idx1) = 0.0082 * (predict_wt(idx1) .^ 3);

% Add random fluctuations
actual_wt = actual_wt .* (1 + noise_intensity*randn(size(actual_wt)));

% Data correction
actual_wt(actual_wt < 0) = 0; % Remove negative values

%% 4. Generate load data

L= 3.7 .* predict_pv + 330 .* actual_wt;
L_rand = 20*sin(2*pi*(1:8760)/8760) + 20*sin(2*pi*(1:8760)/24) + 20*randn(1,8760);
L = L + L_rand';

% Data correction
L(L < 0) = 0;                              % Remove negative values

%% 5. Generate price parameters
season_days = [92, 92, 91, 90];            % Number of days in each season
hourly_mu = readmatrix('electricity price.xlsx'); 
hourly_sigma = 0.1 * hourly_mu;            % Standard deviation

% Generate seasonal typical daily price profiles
price = zeros(8760,1);
start_idx = 1;

for season = 1:4
    % Generate 24-hour normal-distributed prices for one day
    daily_pattern = arrayfun(@(h) normrnd(hourly_mu(h,season),...
                            hourly_sigma(h,season)), 1:24);
    
    % Extend to all days in the season
    season_hours = repmat(daily_pattern, season_days(season), 1);
    end_idx = start_idx + numel(season_hours) -1;
    
    % Fill full-year data
    price(start_idx:end_idx) = season_hours(:);
    start_idx = end_idx + 1;
end

% Add random fluctuations
price = price .* (1 + noise_intensity*randn(size(price)));

% Data correction
price(price < 0) = 0;                               % Remove negative values and enforce non-negative prices
price = min(price, 1.2*max(hourly_mu(:)));          % Limit maximum price

%% Objective function and constraints

% Initialize constraint set
constraints = [];

% Capacity and power constraints of microgrid components
C_rate = 1;                                             
constraints = [constraints, P_bat == E_bat * C_rate];
constraints = [constraints, C_pv <= 4500, C_wt <= 330];       % Upper bounds
constraints = [constraints, E_bat >= 100, E_bat <= 1000];     % Battery capacity bounds              
constraints = [constraints, C_pv >= 0, C_wt >= 100];          % Minimum capacities            

% Land area constraint
constraints = [constraints, C_pv * 5 + C_wt * 12 <= 22400];

% Compute Big-M value (set based on data range)
M_big = 50000;  % Increase Big-M value

% Simplified operational constraints
for t = 1:8760
    % PV and wind output constraints
    constraints = [constraints, p_pv(t) == C_pv * predict_pv(t) / 1000];
    constraints = [constraints, p_wt(t) == C_wt * actual_wt(t)]; 
    
    % Power balance constraints
    constraints = [constraints, p_pv(t) + p_wt(t) + p_bat_dis(t) + p_buy(t) == L(t) + p_bat_ch(t) + mg_load(t)];
    
    % SOC state transition constraints
    if t == 1
        constraints = [constraints, SOC(t) == 0.5*E_bat]; 
    else
        constraints = [constraints, SOC(t) == SOC(t-1) + 0.95*p_bat_ch(t) - p_bat_dis(t)/0.95];
    end
    
    % SOC limits
    constraints = [constraints, SOC(t) >= 0.1*E_bat];    % Minimum SOC 10%
    constraints = [constraints, SOC(t) <= 0.9*E_bat];    % Maximum SOC 90%
    
    % Charging/discharging power constraints
    constraints = [constraints, p_bat_ch(t) <= P_bat, p_bat_ch(t) >= 0];
    constraints = [constraints, p_bat_dis(t) <= P_bat, p_bat_dis(t) >= 0];
    
    % Avoid simultaneous charging/discharging or simultaneous buying/selling
    theta(t) = binvar(1,1);
    constraints = [constraints, p_bat_ch(t) <= M_big * theta(t)];
    constraints = [constraints, p_bat_dis(t) <= M_big * (1 - theta(t))];
    constraints = [constraints, mg_load(t) <= M_big * theta(t)];
    constraints = [constraints, p_buy(t) <= M_big * (1 - theta(t))];
    
    % Grid purchase and sale constraints
    constraints = [constraints, p_buy(t) >= 0];
    constraints = [constraints, mg_load(t) >= 0];
end

% Annual grid purchase ratio constraint
constraints = [constraints, sum(p_buy) <= 0.3 * sum(L)];   

% Objective function: annual net profit
revenue = sum(price .* L)/1000;                               % Revenue from selling electricity
savings = price_mg * sum(mg_load);                            % Savings from reduced grid purchases
cost_grid = 1.2 * sum(price .* p_buy)/1000;                   % Cost of purchased electricity from the grid
cost_cap = 300*C_pv + 900*C_wt + 75*E_bat;                    % Capital cost of equipment
Profit = (revenue + savings - cost_grid - cost_cap) / 10000;  % Convert to units of 10,000

%% Solver settings
fprintf('Start solving optimization problem...\n');

options = sdpsettings('solver', 'cplex', 'verbose', 1, ...
                         'cplex.timelimit', 3600, ...
                         'cplex.mip.tolerances.mipgap', 0.05);

% Solve
result = optimize(constraints, -Profit, options);  

% Check solver status and print details
fprintf('Solver status: %d\n', result.problem);
fprintf('Solver info: %s\n', result.info);

if result.problem == 0
    fprintf('Optimization successful!\n');
    
    % Retrieve decision variable values
    pv_val = value(C_pv);
    wt_val = value(C_wt);
    E_val = value(E_bat);
    Pro_val = value(Profit);

    % Retrieve operation variable values
    p_pv_val = value(p_pv);
    p_wt_val = value(p_wt);
    p_bat_ch_val = value(p_bat_ch);
    p_bat_dis_val = value(p_bat_dis);
    p_buy_val = value(p_buy);
    mg_load_val = value(mg_load);
    price_val = value(price);
    
    % Display results
    fprintf('Optimized capacity configuration:\n');
    fprintf('PV capacity: %.2f kW\n', pv_val);
    fprintf('Wind turbine capacity: %.2f kW\n', wt_val);
    fprintf('Battery capacity: %.2f kWh\n', E_val);
    fprintf('Annual net profit: %.2f (10k units)\n', Pro_val);
    
    % Verify constraint satisfaction
    fprintf('\nConstraint check:\n');
    fprintf('Land area utilization: %.2f%%\n', (pv_val*5 + wt_val*12)/22400*100);
    fprintf('Grid purchase ratio: %.2f%%\n', sum(value(p_buy))/sum(L)*100);

    % Save results to files
    fprintf('\nSaving results to files...\n');
    
    % Create results struct
    results.capacity.pv = pv_val;
    results.capacity.wt = wt_val;
    results.capacity.battery = E_val;
    results.profit = Pro_val;

    results.operation.pv_output = p_pv_val;
    results.operation.wt_output = p_wt_val;
    results.operation.bat_charge = p_bat_ch_val;
    results.operation.bat_discharge = p_bat_dis_val;
    results.operation.grid_purchase = p_buy_val;
    results.operation.mg_load = mg_load_val;
    results.operation.L = L;
    results.operation.price = price_val;

    
    % Save to MAT file
    save('microgrid_optimization_results.mat', 'results');
    
    % Save operation data to CSV file
    operation_table = table((1:8760)', p_pv_val, p_wt_val, p_bat_ch_val, p_bat_dis_val, ...
                           p_buy_val, mg_load_val, L, price_val, ...
                           'VariableNames', {'Hour', 'PV_Output', 'WT_Output', 'Bat_Charge', ...
                           'Bat_Discharge', 'Grid_Purchase', 'MG_Load', 'Load', 'price'});
    writetable(operation_table, 'microgrid_operation_data.csv');
    
    fprintf('Results saved to:\n');
    fprintf('  - microgrid_optimization_results.mat (MATLAB format)\n');
    fprintf('  - microgrid_operation_data.csv (CSV format)\n');

else
    fprintf('Optimization failed, problem code: %d\n', result.problem);
    fprintf('Suggestions:\n');
    fprintf('1. Check data validity\n');
    fprintf('2. Relax constraints further\n');
    fprintf('3. Adjust Big-M value\n');
    fprintf('4. Check model feasibility\n');
    
    % Try feasibility check
    fprintf('\nRunning feasibility check...\n');
    feasibility_result = optimize(constraints, 0, options);
    if feasibility_result.problem == 0
        fprintf('Constraint set is feasible; issue may be in the objective function\n');
    else
        fprintf('Constraint set is infeasible; constraints need to be relaxed\n');
    end
end

% Clear YALMIP variables
yalmip('clear');

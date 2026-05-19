%% unified_stock_selection_and_portfolio_model.m
clear; clc; close all;

%% ============================ 0. 参数区 ============================
SCRIPT_DIR = fileparts(mfilename('fullpath'));
INPUT_CSV = fullfile(SCRIPT_DIR, 'all_50_stocks_2019_2025_baostock.csv');
OUTPUT_DIR = fullfile(SCRIPT_DIR, 'unified_portfolio_matlab_outputs');

if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

% 数据口径
ANALYSIS_YEARS = 3;
FRONT_ADJUST_FLAG = 2;
USE_FRONT_ADJUSTED_CLOSE = true;
SUSPENSION_DAYS_THRESHOLD = 30;
RETURN_METHOD = "log";
ANNUALIZATION = 252;

% 聚类选股
K_MIN = 2;
K_MAX = 8;
FINAL_STOCK_COUNT = 4;
CORRELATION_THRESHOLD = 0.70;
MAX_CORRELATION_REVIEW_ITERATIONS = 50;

% 轮廓系数与 CH 指标联合评分
SILHOUETTE_WEIGHT = 0.50;
CH_WEIGHT = 0.50;

% 类内个股收益风险比
STOCK_SCORE_RISK_FREE_ANNUAL = 0.0;

% 统一主模型
LOWER_WEIGHT = 0.05;
UPPER_WEIGHT = 0.40;
PAIR_WEIGHT_UPPER = 0.65;
TARGET_ANNUAL_RETURN = 0.15;
RISK_FREE_ANNUAL = 0.0;

% lambda 扫描
LAMBDA_START = 0.0001;
LAMBDA_END = 0.1;
LAMBDA_STEP = 0.0001;

% 绘图参数
PLOT_DPI = 300;

fprintf('====================================================================================================\n');
fprintf('Unified Stock Selection + Portfolio Optimization Pipeline (MATLAB)\n');
fprintf('====================================================================================================\n');
fprintf('Input CSV : %s\n', INPUT_CSV);
fprintf('Output Dir: %s\n\n', OUTPUT_DIR);

%% ============================ 1. 读取数据 ============================
if ~isfile(INPUT_CSV)
    error('未找到输入文件：%s', INPUT_CSV);
end

Traw = readtable(INPUT_CSV, 'TextType', 'string', 'VariableNamingRule', 'preserve');
rawRows = height(Traw);
varNames = string(Traw.Properties.VariableNames);

codeCol = firstExistingColumn(varNames, ["股票代码", "stock_code", "证券代码", "code"], "股票代码");
dateCol = firstExistingColumn(varNames, ["date", "日期", "trade_date", "交易日期"], "交易日期");
closeCol = firstExistingColumn(varNames, ["close", "收盘", "close_price", "收盘价"], "收盘价");
adjustCol = optionalColumn(varNames, ["adjustflag", "adjust_flag", "复权标志"]);
tradeStatusCol = optionalColumn(varNames, ["tradestatus", "trade_status", "交易状态"]);

stockCode = normalizeStockCodeColumn(Traw.(codeCol));
tradeDate = parseDateColumn(Traw.(dateCol));
closePrice = toDoubleColumn(Traw.(closeCol));

if strlength(adjustCol) > 0
    adjustFlag = toDoubleColumn(Traw.(adjustCol));
else
    adjustFlag = NaN(rawRows, 1);
end

if strlength(tradeStatusCol) > 0
    tradeStatus = toDoubleColumn(Traw.(tradeStatusCol));
else
    tradeStatus = ones(rawRows, 1);
end

%% ============================ 2. 最近三年窗口 ============================
maxDate = max(tradeDate);
if isnat(maxDate)
    error('数据中没有有效交易日期。');
end

analysisStart = datetime(year(maxDate) - ANALYSIS_YEARS + 1, 1, 1);
analysisEnd = dateshift(maxDate, 'start', 'day');

windowMask = tradeDate >= analysisStart & tradeDate <= analysisEnd;
stockCode = stockCode(windowMask);
tradeDate = tradeDate(windowMask);
closePrice = closePrice(windowMask);
adjustFlag = adjustFlag(windowMask);
tradeStatus = tradeStatus(windowMask);
windowRows = sum(windowMask);

%% ============================ 3. 使用前复权收盘价 ============================
if USE_FRONT_ADJUSTED_CLOSE && strlength(adjustCol) > 0
    frontMask = adjustFlag == FRONT_ADJUST_FLAG;
else
    frontMask = true(size(stockCode));
end

stockCode = stockCode(frontMask);
tradeDate = tradeDate(frontMask);
closePrice = closePrice(frontMask);
adjustFlag = adjustFlag(frontMask);
tradeStatus = tradeStatus(frontMask);
adjustedRows = sum(frontMask);

%% ============================ 4. 基础有效性清洗 ============================
basicValid = ...
    stockCode ~= "" & ...
    ~isnat(tradeDate) & ...
    ~isnan(closePrice) & ...
    isfinite(closePrice) & ...
    closePrice > 0;

S = table( ...
    stockCode(basicValid), ...
    tradeDate(basicValid), ...
    closePrice(basicValid), ...
    adjustFlag(basicValid), ...
    tradeStatus(basicValid), ...
    'VariableNames', {'stock_code', 'trade_date', 'close', 'adjustflag', 'tradestatus'} ...
);

validRowsBeforeSuspension = height(S);

%% ============================ 5. 停牌天数统计与筛除 ============================
S.is_suspended = double(S.tradestatus ~= 1);
[Gstock, stockLevels] = findgroups(S.stock_code);

suspensionDays = splitapply(@sum, S.is_suspended, Gstock);
sampleRows = splitapply(@numel, S.trade_date, Gstock);
sampleStart = splitapply(@min, S.trade_date, Gstock);
sampleEnd = splitapply(@max, S.trade_date, Gstock);

suspensionSummary = table( ...
    stockLevels, suspensionDays, sampleRows, sampleStart, sampleEnd, ...
    'VariableNames', {'stock_code', 'suspension_days', 'sample_rows', 'sample_start', 'sample_end'} ...
);
suspensionSummary = sortrows(suspensionSummary, {'suspension_days', 'stock_code'}, {'descend', 'ascend'});

excludedSuspensionCodes = suspensionSummary.stock_code( ...
    suspensionSummary.suspension_days > SUSPENSION_DAYS_THRESHOLD ...
);

if ~isempty(excludedSuspensionCodes)
    S = S(~ismember(S.stock_code, excludedSuspensionCodes), :);
end

% 仅保留正常交易日
S = S(S.tradestatus == 1, :);
validRowsAfterSuspension = height(S);

%% ============================ 6. 去重并构造价格矩阵 ============================
S = sortrows(S, {'stock_code', 'trade_date'});
duplicateKey = strcat(S.stock_code, "_", string(S.trade_date, 'yyyy-MM-dd'));
[~, keepIdx] = unique(duplicateKey, 'last');
S = S(sort(keepIdx), :);

allDates = unique(S.trade_date);
allCodes = unique(S.stock_code);

nDates = numel(allDates);
nCodes = numel(allCodes);

priceMat = NaN(nDates, nCodes);
[~, dateIdx] = ismember(S.trade_date, allDates);
[~, codeIdx] = ismember(S.stock_code, allCodes);
linIdx = sub2ind(size(priceMat), dateIdx, codeIdx);
priceMat(linIdx) = S.close;

commonMask = all(~isnan(priceMat), 2);
alignedPrices = priceMat(commonMask, :);
alignedDates = allDates(commonMask);

if size(alignedPrices, 1) < 2
    error('共同交易日数量不足，无法计算收益率。');
end

%% ============================ 7. 对数收益率 ============================
switch RETURN_METHOD
    case "log"
        returnsMat = diff(log(alignedPrices), 1, 1);
    case "simple"
        returnsMat = alignedPrices(2:end, :) ./ alignedPrices(1:end-1, :) - 1;
    otherwise
        error('RETURN_METHOD 只能为 "log" 或 "simple"。');
end

returnDates = alignedDates(2:end);

if isempty(returnsMat)
    error('收益率矩阵为空。');
end

%% ============================ 8. 候选股基础统计量 ============================
dailyMean = mean(returnsMat, 1)';
dailyVol = std(returnsMat, 0, 1)';
annualReturn = dailyMean * ANNUALIZATION;
annualVolatility = dailyVol * sqrt(ANNUALIZATION);
stockScore = (annualReturn - STOCK_SCORE_RISK_FREE_ANNUAL) ./ annualVolatility;

candidateStats = table( ...
    allCodes, dailyMean, dailyVol, annualReturn, annualVolatility, stockScore, ...
    'VariableNames', { ...
        'stock_code', 'daily_mean_return', 'daily_volatility', ...
        'annual_return', 'annual_volatility', 'stock_score' ...
    } ...
);
candidateStats = sortrows(candidateStats, ...
    {'stock_score', 'annual_return', 'annual_volatility'}, ...
    {'descend', 'descend', 'ascend'});

%% ============================ 9. 相关距离 + 层次聚类 ============================
corrMat = corrcoef(returnsMat);
corrMat = max(min(corrMat, 1), -1);

distanceMat = sqrt(2 * (1 - corrMat));
distanceMat(1:size(distanceMat,1)+1:end) = 0;

distanceVec = squareform(distanceMat);
Z = linkage(distanceVec, 'average');

% 标准化收益率特征，用于 CH 指标
features = standardizeFeatureRows(returnsMat');

kUpper = min(K_MAX, nCodes - 1);
if kUpper < K_MIN
    error('候选股票数量不足，无法搜索聚类数。');
end

kList = (K_MIN:kUpper)';
silhouetteScore = zeros(numel(kList), 1);
chScore = zeros(numel(kList), 1);

for idx = 1:numel(kList)
    k = kList(idx);
    labelsK = cluster(Z, 'Maxclust', k);
    silhouetteScore(idx) = silhouetteFromDistance(distanceMat, labelsK);
    chScore(idx) = calinskiHarabaszManual(features, labelsK);
end

silhouetteNorm = minmaxNormalize(silhouetteScore);
chNorm = minmaxNormalize(chScore);
jointScore = SILHOUETTE_WEIGHT * silhouetteNorm + CH_WEIGHT * chNorm;

[~, bestIdx] = max(jointScore);
optimalK = kList(bestIdx);

kMetrics = table( ...
    kList, silhouetteScore, chScore, silhouetteNorm, chNorm, jointScore, ...
    'VariableNames', { ...
        'k', 'silhouette_score', 'ch_score', ...
        'silhouette_norm', 'ch_norm', 'joint_score' ...
    } ...
);

clusterIds = cluster(Z, 'Maxclust', optimalK);
clusterAssignments = table(allCodes, clusterIds, ...
    'VariableNames', {'stock_code', 'cluster_id'});
clusterAssignments = sortrows(clusterAssignments, {'cluster_id', 'stock_code'});

%% ============================ 10. 聚类摘要与候选排序 ============================
clusterSummary = buildClusterSummary(clusterAssignments, candidateStats, distanceMat, allCodes);
clusterCandidates = buildClusterCandidateCells(clusterAssignments, candidateStats);

%% ============================ 11. 按统一模型规则选出最终 4 股 ============================
[selectedStocks, selectionReviewLog] = selectFinalFourStocks( ...
    optimalK, FINAL_STOCK_COUNT, clusterSummary, clusterCandidates, ...
    candidateStats, corrMat, allCodes, CORRELATION_THRESHOLD, ...
    MAX_CORRELATION_REVIEW_ITERATIONS ...
);

selectedCodes = selectedStocks.stock_code;
[~, selectedColIdx] = ismember(selectedCodes, allCodes);

selectedReturns = returnsMat(:, selectedColIdx);
selectedPrices = alignedPrices(:, selectedColIdx);
selectedPairCorr = corrcoef(selectedReturns);

%% ============================ 12. 年化 mu 与 Sigma ============================
muAnnual = mean(selectedReturns, 1)' * ANNUALIZATION;
covAnnual = cov(selectedReturns) * ANNUALIZATION;

portfolioInputSummary = table( ...
    selectedCodes, ...
    muAnnual, ...
    sqrt(diag(covAnnual)), ...
    (muAnnual - RISK_FREE_ANNUAL) ./ sqrt(diag(covAnnual)), ...
    'VariableNames', { ...
        'stock_code', 'annual_expected_return', ...
        'annual_volatility', 'single_asset_sharpe_like' ...
    } ...
);

%% ============================ 13. 统一结构约束 ============================
nAssets = numel(selectedCodes);
Aeq = ones(1, nAssets);
beq = 1;

lb = LOWER_WEIGHT * ones(nAssets, 1);
ub = UPPER_WEIGHT * ones(nAssets, 1);

[A_pair, b_pair] = pairwiseConstraintMatrix(nAssets, PAIR_WEIGHT_UPPER);

% 可行收益区间
lpOptions = optimoptions('linprog', 'Display', 'none');

wMinReturn = linprog( ...
    muAnnual, A_pair, b_pair, Aeq, beq, lb, ub, lpOptions ...
);
wMaxReturn = linprog( ...
    -muAnnual, A_pair, b_pair, Aeq, beq, lb, ub, lpOptions ...
);

minFeasibleReturn = muAnnual' * wMinReturn;
maxFeasibleReturn = muAnnual' * wMaxReturn;

feasibleReturnRanges = table( ...
    "unified_structural_constraints", ...
    LOWER_WEIGHT, ...
    UPPER_WEIGHT, ...
    PAIR_WEIGHT_UPPER, ...
    minFeasibleReturn, ...
    maxFeasibleReturn, ...
    TARGET_ANNUAL_RETURN, ...
    string(ternary(TARGET_ANNUAL_RETURN <= maxFeasibleReturn + 1e-9, 'yes', 'no')), ...
    'VariableNames', { ...
        'constraint_case', 'lower_weight', 'upper_weight', 'pair_weight_upper', ...
        'min_feasible_annual_return', 'max_feasible_annual_return', ...
        'target_annual_return_R0', 'target_return_feasible' ...
    } ...
);

if TARGET_ANNUAL_RETURN > maxFeasibleReturn + 1e-9
    error('R0=%.4f 超出最大可行收益 %.4f，请降低 R0。', ...
        TARGET_ANNUAL_RETURN, maxFeasibleReturn);
end

%% ============================ 14. 统一主模型求解 ============================
qpOptions = optimoptions('quadprog', 'Display', 'none');

H = 2 * covAnnual;

% 14.1 等权重组合
wEqual = ones(nAssets, 1) / nAssets;

% 14.2 最小风险组合：lambda=0，无收益下界
lambdaMinRisk = 0.0;
fMinRisk = -lambdaMinRisk * muAnnual;
wMinimumRisk = quadprog( ...
    H, fMinRisk, A_pair, b_pair, Aeq, beq, lb, ub, [], qpOptions ...
);

% 14.3 目标收益约束最小风险组合：lambda=0，含收益下界
A_target = [A_pair; -muAnnual'];
b_target = [b_pair; -TARGET_ANNUAL_RETURN];
wTargetMinRisk = quadprog( ...
    H, fMinRisk, A_target, b_target, Aeq, beq, lb, ub, [], qpOptions ...
);

% 14.4 lambda 扫描有效前沿，不含收益下界
lambdaValues = (LAMBDA_START:LAMBDA_STEP:LAMBDA_END)';
nLambda = numel(lambdaValues);

frontierWeights = NaN(nLambda, nAssets);
frontierReturn = NaN(nLambda, 1);
frontierVariance = NaN(nLambda, 1);
frontierVolatility = NaN(nLambda, 1);
frontierSharpe = NaN(nLambda, 1);
frontierMaxSingleWeight = NaN(nLambda, 1);
frontierMaxPairWeightSum = NaN(nLambda, 1);

for idx = 1:nLambda
    lambdaValue = lambdaValues(idx);
    fLambda = -lambdaValue * muAnnual;

    [wLambda, ~, exitflag] = quadprog( ...
        H, fLambda, A_pair, b_pair, Aeq, beq, lb, ub, [], qpOptions ...
    );

    if exitflag <= 0 || isempty(wLambda)
        continue;
    end

    frontierWeights(idx, :) = wLambda';
    frontierReturn(idx) = muAnnual' * wLambda;
    frontierVariance(idx) = wLambda' * covAnnual * wLambda;
    frontierVolatility(idx) = sqrt(max(frontierVariance(idx), 0));
    frontierSharpe(idx) = portfolioSharpe(wLambda, muAnnual, covAnnual, RISK_FREE_ANNUAL);
    frontierMaxSingleWeight(idx) = max(wLambda);
    frontierMaxPairWeightSum(idx) = maxPairWeightSum(wLambda);
end

validFrontier = ~isnan(frontierSharpe);
lambdaValuesValid = lambdaValues(validFrontier);
frontierWeightsValid = frontierWeights(validFrontier, :);
frontierReturnValid = frontierReturn(validFrontier);
frontierVarianceValid = frontierVariance(validFrontier);
frontierVolatilityValid = frontierVolatility(validFrontier);
frontierSharpeValid = frontierSharpe(validFrontier);
frontierMaxSingleWeightValid = frontierMaxSingleWeight(validFrontier);
frontierMaxPairWeightSumValid = frontierMaxPairWeightSum(validFrontier);

[~, tangentIdx] = max(frontierSharpeValid);
lambdaStar = lambdaValuesValid(tangentIdx);
wTangency = frontierWeightsValid(tangentIdx, :)';

%% ============================ 15. 组合对比表 ============================
portfolioNames = [ ...
    "Equal Weight"; ...
    "Minimum Risk"; ...
    "Target-Return-Constrained Minimum Risk"; ...
    "Optimal Tangency Portfolio" ...
];

portfolioLambda = [NaN; 0.0; 0.0; lambdaStar];
containsTargetReturnConstraint = ["no"; "no"; "yes"; "no"];

W = [ ...
    wEqual'; ...
    wMinimumRisk'; ...
    wTargetMinRisk'; ...
    wTangency' ...
];

nPortfolios = size(W, 1);
annualExpectedReturn = zeros(nPortfolios, 1);
annualVariance = zeros(nPortfolios, 1);
annualVolatility = zeros(nPortfolios, 1);
sharpeRatio = zeros(nPortfolios, 1);
maxSingleWeight = zeros(nPortfolios, 1);
maxPairWeightSumVec = zeros(nPortfolios, 1);

for p = 1:nPortfolios
    wp = W(p, :)';
    annualExpectedReturn(p) = muAnnual' * wp;
    annualVariance(p) = wp' * covAnnual * wp;
    annualVolatility(p) = sqrt(max(annualVariance(p), 0));
    sharpeRatio(p) = portfolioSharpe(wp, muAnnual, covAnnual, RISK_FREE_ANNUAL);
    maxSingleWeight(p) = max(wp);
    maxPairWeightSumVec(p) = maxPairWeightSum(wp);
end

portfolioComparison = table( ...
    portfolioNames, ...
    portfolioLambda, ...
    containsTargetReturnConstraint, ...
    annualExpectedReturn, ...
    annualVariance, ...
    annualVolatility, ...
    sharpeRatio, ...
    maxSingleWeight, ...
    maxPairWeightSumVec, ...
    'VariableNames', { ...
        'portfolio', 'lambda', 'contains_target_return_constraint', ...
        'annual_expected_return', 'annual_variance', 'annual_volatility', ...
        'sharpe_ratio', 'max_single_weight', 'max_pair_weight_sum' ...
    } ...
);

for j = 1:nAssets
    portfolioComparison.(sprintf('weight_%s', selectedCodes(j))) = W(:, j);
end

%% ============================ 16. lambda 前沿表 ============================
lambdaFrontier = table( ...
    lambdaValuesValid, ...
    frontierReturnValid, ...
    frontierVarianceValid, ...
    frontierVolatilityValid, ...
    frontierSharpeValid, ...
    frontierMaxSingleWeightValid, ...
    frontierMaxPairWeightSumValid, ...
    'VariableNames', { ...
        'lambda', 'annual_expected_return', 'annual_variance', ...
        'annual_volatility', 'sharpe_ratio', ...
        'max_single_weight', 'max_pair_weight_sum' ...
    } ...
);

for j = 1:nAssets
    lambdaFrontier.(sprintf('weight_%s', selectedCodes(j))) = frontierWeightsValid(:, j);
end

%% ============================ 17. 输出 CSV ============================
cleaningSummary = table( ...
    rawRows, ...
    windowRows, ...
    adjustedRows, ...
    validRowsBeforeSuspension, ...
    validRowsAfterSuspension, ...
    string(analysisStart, 'yyyy-MM-dd'), ...
    string(analysisEnd, 'yyyy-MM-dd'), ...
    string(strjoin(cellstr(excludedSuspensionCodes), ', ')), ...
    string(alignedDates(1), 'yyyy-MM-dd'), ...
    string(alignedDates(end), 'yyyy-MM-dd'), ...
    size(alignedPrices, 1), ...
    string(returnDates(1), 'yyyy-MM-dd'), ...
    string(returnDates(end), 'yyyy-MM-dd'), ...
    size(returnsMat, 1), ...
    nCodes, ...
    optimalK, ...
    'VariableNames', { ...
        'raw_rows', 'window_rows_recent_3_years', 'rows_after_front_adjust_filter', ...
        'rows_after_basic_validity_check', 'rows_after_suspension_filter_and_normal_trading_filter', ...
        'analysis_start', 'analysis_end', 'excluded_suspension_codes', ...
        'aligned_price_start', 'aligned_price_end', 'aligned_price_days', ...
        'return_start', 'return_end', 'return_days', ...
        'candidate_stock_count_after_suspension_filter', 'optimal_k' ...
    } ...
);

writetable(cleaningSummary, fullfile(OUTPUT_DIR, '00_cleaning_summary.csv'));
writetable(suspensionSummary, fullfile(OUTPUT_DIR, '01_suspension_days_summary.csv'));

alignedPriceTable = array2table(alignedPrices, 'VariableNames', makeSafeVariableNames(allCodes));
alignedPriceTable = addvars(alignedPriceTable, alignedDates, 'Before', 1, 'NewVariableNames', 'trade_date');
writetable(alignedPriceTable, fullfile(OUTPUT_DIR, '02_aligned_front_adjusted_close_prices.csv'));

allReturnTable = array2table(returnsMat, 'VariableNames', makeSafeVariableNames(allCodes));
allReturnTable = addvars(allReturnTable, returnDates, 'Before', 1, 'NewVariableNames', 'trade_date');
writetable(allReturnTable, fullfile(OUTPUT_DIR, '03_all_candidate_log_returns.csv'));

writetable(candidateStats, fullfile(OUTPUT_DIR, '04_candidate_stock_statistics_numeric.csv'));
writetable(kMetrics, fullfile(OUTPUT_DIR, '05_k_selection_metrics.csv'));
writetable(clusterAssignments, fullfile(OUTPUT_DIR, '06_optimal_k_cluster_assignments.csv'));
writetable(clusterSummary, fullfile(OUTPUT_DIR, '07_cluster_summary.csv'));
writetable(selectedStocks, fullfile(OUTPUT_DIR, '08_selected_4_stocks_numeric.csv'));
writetable(selectionReviewLog, fullfile(OUTPUT_DIR, '09_selection_and_correlation_review_log.csv'));

selectedCorrTable = array2table(selectedPairCorr, 'VariableNames', makeSafeVariableNames(selectedCodes));
selectedCorrTable = addvars(selectedCorrTable, selectedCodes, 'Before', 1, 'NewVariableNames', 'stock_code');
writetable(selectedCorrTable, fullfile(OUTPUT_DIR, '10_selected_4_pairwise_correlation_matrix.csv'));

selectedReturnTable = array2table(selectedReturns, 'VariableNames', makeSafeVariableNames(selectedCodes));
selectedReturnTable = addvars(selectedReturnTable, returnDates, 'Before', 1, 'NewVariableNames', 'trade_date');
writetable(selectedReturnTable, fullfile(OUTPUT_DIR, '11_selected_4_log_returns.csv'));

selectedPriceTable = array2table(selectedPrices, 'VariableNames', makeSafeVariableNames(selectedCodes));
selectedPriceTable = addvars(selectedPriceTable, alignedDates, 'Before', 1, 'NewVariableNames', 'trade_date');
writetable(selectedPriceTable, fullfile(OUTPUT_DIR, '12_selected_4_front_adjusted_close_prices.csv'));

annualMeanTable = table(selectedCodes, muAnnual, ...
    'VariableNames', {'stock_code', 'annual_expected_return'});
writetable(annualMeanTable, fullfile(OUTPUT_DIR, '13_selected_4_annual_mean_returns.csv'));

annualCovTable = array2table(covAnnual, 'VariableNames', makeSafeVariableNames(selectedCodes));
annualCovTable = addvars(annualCovTable, selectedCodes, 'Before', 1, 'NewVariableNames', 'stock_code');
writetable(annualCovTable, fullfile(OUTPUT_DIR, '14_selected_4_annual_covariance_matrix.csv'));

writetable(portfolioInputSummary, fullfile(OUTPUT_DIR, '15_selected_4_portfolio_input_summary.csv'));
writetable(feasibleReturnRanges, fullfile(OUTPUT_DIR, '16_feasible_return_range_under_unified_constraints.csv'));
writetable(lambdaFrontier, fullfile(OUTPUT_DIR, '17_lambda_scanned_effective_frontier_numeric.csv'));
writetable(portfolioComparison, fullfile(OUTPUT_DIR, '18_portfolio_comparison_numeric.csv'));

%% ============================ 18. 输出图像 ============================
% 18.1 候选股相关系数热力图
figure('Position', [100, 100, 1150, 900]);
imagesc(corrMat);
colorbar;
title('Correlation Matrix of Candidate Stock Returns');
xlabel('Stock Code');
ylabel('Stock Code');
xticks(1:nCodes);
yticks(1:nCodes);
xticklabels(cellstr(allCodes));
yticklabels(cellstr(allCodes));
xtickangle(90);
exportgraphics(gcf, fullfile(OUTPUT_DIR, '19_candidate_correlation_heatmap.png'), 'Resolution', PLOT_DPI);
close(gcf);

% 18.2 聚类数选择图
figure('Position', [100, 100, 900, 600]);
yyaxis left;
plot(kList, silhouetteScore, '-o', 'LineWidth', 1.5);
ylabel('Silhouette Score');
yyaxis right;
plot(kList, chScore, '-s', 'LineWidth', 1.5);
ylabel('CH Score');
hold on;
xline(optimalK, '--', sprintf('Selected k=%d', optimalK), 'LineWidth', 1.2);
grid on;
xlabel('Number of Clusters k');
title('Cluster Number Selection by Silhouette and CH Indicators');
legend({'Silhouette Score', 'CH Score', 'Selected k'}, 'Location', 'best');
exportgraphics(gcf, fullfile(OUTPUT_DIR, '20_silhouette_ch_cluster_number_selection.png'), 'Resolution', PLOT_DPI);
close(gcf);

% 18.3 树状图
figure('Position', [100, 100, 1500, 800]);
dendrogram(Z, 0, 'Labels', cellstr(allCodes));
title('Average-Linkage Hierarchical Clustering Based on Correlation Distance');
xlabel('Stock Code');
ylabel('Correlation Distance');
xtickangle(90);
exportgraphics(gcf, fullfile(OUTPUT_DIR, '21_hierarchical_clustering_dendrogram.png'), 'Resolution', PLOT_DPI);
close(gcf);

% 18.4 聚类规模柱状图
clusterCountLevels = unique(clusterIds);
clusterCounts = zeros(numel(clusterCountLevels), 1);
for idx = 1:numel(clusterCountLevels)
    clusterCounts(idx) = sum(clusterIds == clusterCountLevels(idx));
end

figure('Position', [100, 100, 750, 500]);
bar(clusterCountLevels, clusterCounts);
grid on;
title('Number of Stocks in Each Cluster');
xlabel('Cluster');
ylabel('Number of Stocks');
exportgraphics(gcf, fullfile(OUTPUT_DIR, '22_optimal_k_cluster_size_bar.png'), 'Resolution', PLOT_DPI);
close(gcf);

% 18.5 最终四股相关性热力图
figure('Position', [100, 100, 650, 550]);
imagesc(selectedPairCorr, [-1, 1]);
colorbar;
title('Pairwise Correlation of Final 4 Selected Stocks');
xlabel('Stock Code');
ylabel('Stock Code');
xticks(1:nAssets);
yticks(1:nAssets);
xticklabels(cellstr(selectedCodes));
yticklabels(cellstr(selectedCodes));
xtickangle(45);
exportgraphics(gcf, fullfile(OUTPUT_DIR, '23_selected_4_pairwise_correlation_heatmap.png'), 'Resolution', PLOT_DPI);
close(gcf);

% 18.6 lambda 扫描有效前沿
figure('Position', [100, 100, 1000, 650]);
plot(frontierVolatilityValid, frontierReturnValid, 'LineWidth', 1.8);
hold on;
scatter(annualVolatility, annualExpectedReturn, 75, 'filled');
grid on;
title('Effective Frontier from Unified Lambda-Weighted Model');
xlabel('Annualized Volatility');
ylabel('Annualized Expected Return');
legend(["Lambda-Scanned Efficient Frontier"; portfolioNames], 'Location', 'best');
exportgraphics(gcf, fullfile(OUTPUT_DIR, '24_lambda_scanned_effective_frontier.png'), 'Resolution', PLOT_DPI);
close(gcf);

% 18.7 组合权重比较
figure('Position', [100, 100, 1150, 650]);
bar(W');
grid on;
title('Portfolio Weights Comparison');
xlabel('Stock Code');
ylabel('Weight');
xticks(1:nAssets);
xticklabels(cellstr(selectedCodes));
legend(cellstr(portfolioNames), 'Location', 'best');
exportgraphics(gcf, fullfile(OUTPUT_DIR, '25_portfolio_weights_comparison.png'), 'Resolution', PLOT_DPI);
close(gcf);

% 18.8 风险收益比较
figure('Position', [100, 100, 900, 600]);
scatter(annualVolatility, annualExpectedReturn, 80, 'filled');
grid on;
title('Risk-Return Comparison');
xlabel('Annualized Volatility');
ylabel('Annualized Expected Return');
for p = 1:nPortfolios
    text(annualVolatility(p), annualExpectedReturn(p), ...
        "  " + portfolioNames(p), 'FontSize', 9);
end
exportgraphics(gcf, fullfile(OUTPUT_DIR, '26_risk_return_comparison.png'), 'Resolution', PLOT_DPI);
close(gcf);

%% ============================ 19. 控制台摘要 ============================
fprintf('====================================================================================================\n');
fprintf('Data Window and Cleaning Summary\n');
fprintf('====================================================================================================\n');
fprintf('Analysis window                    : %s ~ %s\n', char(string(analysisStart, 'yyyy-MM-dd')), char(string(analysisEnd, 'yyyy-MM-dd')));
fprintf('Aligned close-price sample         : %s ~ %s\n', char(string(alignedDates(1), 'yyyy-MM-dd')), char(string(alignedDates(end), 'yyyy-MM-dd')));
fprintf('Aligned close-price days           : %d\n', size(alignedPrices, 1));
fprintf('Return sample                      : %s ~ %s\n', char(string(returnDates(1), 'yyyy-MM-dd')), char(string(returnDates(end), 'yyyy-MM-dd')));
fprintf('Return sample days                 : %d\n', size(returnsMat, 1));
fprintf('Excluded stocks, suspension > 30   : %s\n\n', char(strjoin(cellstr(excludedSuspensionCodes), ', ')));

fprintf('====================================================================================================\n');
fprintf('Cluster Number Selection\n');
fprintf('====================================================================================================\n');
disp(kMetrics);
fprintf('Selected optimal k*: %d\n\n', optimalK);

fprintf('====================================================================================================\n');
fprintf('Final 4 Selected Stocks\n');
fprintf('====================================================================================================\n');
disp(selectedStocks);

fprintf('====================================================================================================\n');
fprintf('Feasible Return Range Under Unified Structural Constraints\n');
fprintf('====================================================================================================\n');
disp(feasibleReturnRanges);

fprintf('====================================================================================================\n');
fprintf('Portfolio Comparison\n');
fprintf('====================================================================================================\n');
disp(portfolioComparison);

fprintf('====================================================================================================\n');
fprintf('Key Output Files\n');
fprintf('====================================================================================================\n');
fprintf('%s\n', fullfile(OUTPUT_DIR, '08_selected_4_stocks_numeric.csv'));
fprintf('%s\n', fullfile(OUTPUT_DIR, '18_portfolio_comparison_numeric.csv'));
fprintf('%s\n', fullfile(OUTPUT_DIR, '17_lambda_scanned_effective_frontier_numeric.csv'));
fprintf('%s\n', fullfile(OUTPUT_DIR, '20_silhouette_ch_cluster_number_selection.png'));
fprintf('%s\n', fullfile(OUTPUT_DIR, '24_lambda_scanned_effective_frontier.png'));
fprintf('%s\n', fullfile(OUTPUT_DIR, '25_portfolio_weights_comparison.png'));
fprintf('====================================================================================================\n');

%% ============================ 本地函数 ============================

function colName = firstExistingColumn(varNames, candidates, label)
    colName = "";
    for i = 1:numel(candidates)
        if any(varNames == candidates(i))
            colName = candidates(i);
            return;
        end
    end
    error('未找到 %s 字段。当前字段为：%s', label, strjoin(cellstr(varNames), ', '));
end

function colName = optionalColumn(varNames, candidates)
    colName = "";
    for i = 1:numel(candidates)
        if any(varNames == candidates(i))
            colName = candidates(i);
            return;
        end
    end
end

function values = toDoubleColumn(rawColumn)
    if isnumeric(rawColumn)
        values = double(rawColumn);
    else
        values = str2double(string(rawColumn));
    end
end

function dates = parseDateColumn(rawColumn)
    if isdatetime(rawColumn)
        dates = rawColumn;
        return;
    end

    rawStrings = string(rawColumn);
    dates = NaT(numel(rawStrings), 1);

    for i = 1:numel(rawStrings)
        s = strtrim(rawStrings(i));
        if s == "" || ismissing(s)
            continue;
        end

        try
            dates(i) = datetime(s, 'InputFormat', 'yyyy-MM-dd');
        catch
            try
                dates(i) = datetime(s);
            catch
                dates(i) = NaT;
            end
        end
    end
end

function codes = normalizeStockCodeColumn(rawColumn)
    n = numel(rawColumn);
    codes = strings(n, 1);

    if isnumeric(rawColumn)
        for i = 1:n
            if isnan(rawColumn(i))
                codes(i) = "";
            else
                codes(i) = string(sprintf('%06d', round(rawColumn(i))));
            end
        end
        return;
    end

    rawStrings = string(rawColumn);
    for i = 1:n
        s = strtrim(rawStrings(i));
        if s == "" || ismissing(s)
            codes(i) = "";
            continue;
        end

        token = regexp(char(s), '(\d{6})$', 'tokens', 'once');
        if ~isempty(token)
            codes(i) = string(token{1});
            continue;
        end

        numericValue = str2double(s);
        if ~isnan(numericValue)
            codes(i) = string(sprintf('%06d', round(numericValue)));
        else
            codes(i) = s;
        end
    end
end

function normalized = minmaxNormalize(values)
    vmin = min(values);
    vmax = max(values);
    if isnan(vmin) || isnan(vmax) || abs(vmax - vmin) < 1e-12
        normalized = zeros(size(values));
    else
        normalized = (values - vmin) / (vmax - vmin);
    end
end

function features = standardizeFeatureRows(X)
    features = X;
    for i = 1:size(features, 1)
        row = features(i, :);
        row = row - mean(row);
        sd = std(row, 0, 2);
        if sd > 1e-12
            row = row / sd;
        else
            row = zeros(size(row));
        end
        features(i, :) = row;
    end
end

function score = silhouetteFromDistance(distanceMat, labels)
    labels = labels(:);
    n = numel(labels);
    uniqueLabels = unique(labels);
    s = zeros(n, 1);

    for i = 1:n
        sameMask = labels == labels(i);
        sameMask(i) = false;

        if sum(sameMask) == 0
            s(i) = 0;
            continue;
        end

        a_i = mean(distanceMat(i, sameMask));

        b_i = inf;
        for j = 1:numel(uniqueLabels)
            label = uniqueLabels(j);
            if label == labels(i)
                continue;
            end
            otherMask = labels == label;
            if sum(otherMask) > 0
                b_i = min(b_i, mean(distanceMat(i, otherMask)));
            end
        end

        denom = max(a_i, b_i);
        if denom <= 1e-12 || ~isfinite(denom)
            s(i) = 0;
        else
            s(i) = (b_i - a_i) / denom;
        end
    end

    score = mean(s);
end

function score = calinskiHarabaszManual(features, labels)
    labels = labels(:);
    uniqueLabels = unique(labels);
    n = size(features, 1);
    k = numel(uniqueLabels);

    if k <= 1 || n <= k
        score = NaN;
        return;
    end

    overallMean = mean(features, 1);
    betweenDispersion = 0;
    withinDispersion = 0;

    for idx = 1:numel(uniqueLabels)
        label = uniqueLabels(idx);
        clusterX = features(labels == label, :);
        if isempty(clusterX)
            continue;
        end
        clusterMean = mean(clusterX, 1);
        betweenDispersion = betweenDispersion + size(clusterX, 1) * sum((clusterMean - overallMean).^2);
        withinDispersion = withinDispersion + sum(sum((clusterX - clusterMean).^2));
    end

    if withinDispersion <= 1e-12
        score = NaN;
    else
        score = (betweenDispersion / (k - 1)) / (withinDispersion / (n - k));
    end
end

function clusterSummary = buildClusterSummary(clusterAssignments, candidateStats, distanceMat, allCodes)
    clusterIds = unique(clusterAssignments.cluster_id);
    nClusters = numel(clusterIds);

    clusterIdCol = zeros(nClusters, 1);
    clusterSize = zeros(nClusters, 1);
    bestStockScore = zeros(nClusters, 1);
    meanStockScore = zeros(nClusters, 1);
    internalMeanDistance = zeros(nClusters, 1);

    for idx = 1:nClusters
        c = clusterIds(idx);
        codesInCluster = clusterAssignments.stock_code(clusterAssignments.cluster_id == c);
        clusterIdCol(idx) = c;
        clusterSize(idx) = numel(codesInCluster);

        [~, statsIdx] = ismember(codesInCluster, candidateStats.stock_code);
        scores = candidateStats.stock_score(statsIdx);
        bestStockScore(idx) = max(scores);
        meanStockScore(idx) = mean(scores);

        [~, codeIdx] = ismember(codesInCluster, allCodes);
        if numel(codeIdx) >= 2
            subDist = distanceMat(codeIdx, codeIdx);
            triMask = triu(true(size(subDist)), 1);
            internalMeanDistance(idx) = mean(subDist(triMask));
        else
            internalMeanDistance(idx) = 0;
        end
    end

    clusterSummary = table( ...
        clusterIdCol, clusterSize, bestStockScore, meanStockScore, internalMeanDistance, ...
        'VariableNames', { ...
            'cluster_id', 'cluster_size', 'best_stock_score', ...
            'mean_stock_score', 'internal_mean_distance' ...
        } ...
    );
end

function clusterCandidates = buildClusterCandidateCells(clusterAssignments, candidateStats)
    clusterIds = unique(clusterAssignments.cluster_id);
    maxClusterId = max(clusterIds);
    clusterCandidates = cell(maxClusterId, 1);

    for idx = 1:numel(clusterIds)
        c = clusterIds(idx);
        codesInCluster = clusterAssignments.stock_code(clusterAssignments.cluster_id == c);
        [~, statsIdx] = ismember(codesInCluster, candidateStats.stock_code);
        subStats = candidateStats(statsIdx, :);
        subStats = sortrows(subStats, ...
            {'stock_score', 'annual_return', 'annual_volatility'}, ...
            {'descend', 'descend', 'ascend'});
        clusterCandidates{c} = subStats.stock_code;
    end
end

function [selectedStocks, reviewLog] = selectFinalFourStocks( ...
    optimalK, finalStockCount, clusterSummary, clusterCandidates, ...
    candidateStats, corrMat, allCodes, corrThreshold, maxIterations ...
)
    logAction = strings(0, 1);
    logDetail = strings(0, 1);

    if optimalK == finalStockCount
        participatingClusters = clusterSummary.cluster_id;
        rule = "optimal_k_equals_4";
    elseif optimalK > finalStockCount
        ordered = sortrows(clusterSummary, ...
            {'best_stock_score', 'mean_stock_score', 'internal_mean_distance'}, ...
            {'descend', 'descend', 'descend'});
        participatingClusters = ordered.cluster_id(1:finalStockCount);
        rule = "optimal_k_greater_than_4_select_top_4_clusters";
    else
        participatingClusters = clusterSummary.cluster_id;
        rule = "optimal_k_less_than_4_supplement_later";
    end

    logAction(end+1, 1) = "participating_clusters_determined";
    logDetail(end+1, 1) = "rule=" + rule + "; clusters=" + strjoin(string(participatingClusters'), ",");

    selectedCodes = strings(0, 1);
    selectedClusters = zeros(0, 1);
    selectionReason = strings(0, 1);

    for idx = 1:numel(participatingClusters)
        c = participatingClusters(idx);
        topCode = clusterCandidates{c}(1);
        selectedCodes(end+1, 1) = topCode;
        selectedClusters(end+1, 1) = c;
        selectionReason(end+1, 1) = "top_score_in_cluster";
    end

    % 若 k*<4，补选
    if numel(selectedCodes) < finalStockCount
        orderedSupplementClusters = sortrows(clusterSummary, ...
            {'cluster_size', 'internal_mean_distance', 'best_stock_score'}, ...
            {'descend', 'descend', 'descend'});
        supplementClusters = orderedSupplementClusters.cluster_id;

        while numel(selectedCodes) < finalStockCount
            added = false;

            for idx = 1:numel(supplementClusters)
                c = supplementClusters(idx);
                candidates = clusterCandidates{c};

                for j = 1:numel(candidates)
                    candidate = candidates(j);
                    if any(selectedCodes == candidate)
                        continue;
                    end

                    maxCorr = candidateMaxCorrelation(candidate, selectedCodes, corrMat, allCodes);
                    if maxCorr <= corrThreshold + 1e-12
                        selectedCodes(end+1, 1) = candidate;
                        selectedClusters(end+1, 1) = c;
                        selectionReason(end+1, 1) = "supplement_k_less_than_4";

                        logAction(end+1, 1) = "supplement";
                        logDetail(end+1, 1) = ...
                            "added=" + candidate + "; cluster=" + string(c) + "; max_corr=" + string(maxCorr);

                        added = true;
                        break;
                    end
                end

                if added
                    break;
                end
            end

            if ~added
                error('最优类别数小于 4，但无法在相关性阈值约束下补足 4 只股票。');
            end
        end
    end

    % 相关性复核
    for iter = 1:maxIterations
        [maxCorr, pairCodes] = selectedMaxPairCorrelation(selectedCodes, corrMat, allCodes);

        if maxCorr <= corrThreshold + 1e-12
            logAction(end+1, 1) = "correlation_review_passed";
            logDetail(end+1, 1) = "iteration=" + string(iter) + "; max_corr=" + string(maxCorr);
            break;
        end

        bestFound = false;
        bestTrialMaxCorr = inf;
        bestReplaceCode = "";
        bestNewCode = "";
        bestCluster = NaN;

        offendingCodes = pairCodes;

        for p = 1:numel(offendingCodes)
            replaceCode = offendingCodes(p);
            replaceIdx = find(selectedCodes == replaceCode, 1);
            clusterId = selectedClusters(replaceIdx);
            candidates = clusterCandidates{clusterId};

            for j = 1:numel(candidates)
                candidate = candidates(j);
                if any(selectedCodes == candidate)
                    continue;
                end

                trialCodes = selectedCodes;
                trialCodes(replaceIdx) = candidate;
                [trialMaxCorr, ~] = selectedMaxPairCorrelation(trialCodes, corrMat, allCodes);

                if trialMaxCorr < bestTrialMaxCorr
                    bestFound = true;
                    bestTrialMaxCorr = trialMaxCorr;
                    bestReplaceCode = replaceCode;
                    bestNewCode = candidate;
                    bestCluster = clusterId;
                end
            end
        end

        if ~bestFound || bestTrialMaxCorr >= maxCorr - 1e-12
            error('相关性复核失败：类别内部备选股票无法有效降低最大相关系数。');
        end

        replaceIdx = find(selectedCodes == bestReplaceCode, 1);
        selectedCodes(replaceIdx) = bestNewCode;
        selectionReason(replaceIdx) = "correlation_review_replacement";

        logAction(end+1, 1) = "replace_due_to_high_correlation";
        logDetail(end+1, 1) = ...
            "replace=" + bestReplaceCode + "; new=" + bestNewCode + ...
            "; cluster=" + string(bestCluster) + "; new_max_corr=" + string(bestTrialMaxCorr);

        if iter == maxIterations
            error('相关性复核超过最大迭代次数。');
        end
    end

    [~, statsIdx] = ismember(selectedCodes, candidateStats.stock_code);
    selectedStats = candidateStats(statsIdx, :);

    selectedStocks = table( ...
        selectedCodes, selectedClusters, selectionReason, ...
        selectedStats.annual_return, selectedStats.annual_volatility, selectedStats.stock_score, ...
        'VariableNames', { ...
            'stock_code', 'cluster_id', 'selection_reason', ...
            'annual_return', 'annual_volatility', 'stock_score' ...
        } ...
    );
    selectedStocks = sortrows(selectedStocks, {'cluster_id', 'stock_score'}, {'ascend', 'descend'});

    reviewLog = table(logAction, logDetail, ...
        'VariableNames', {'action', 'detail'});
end

function maxCorr = candidateMaxCorrelation(candidate, selectedCodes, corrMat, allCodes)
    if isempty(selectedCodes)
        maxCorr = -inf;
        return;
    end

    [~, candIdx] = ismember(candidate, allCodes);
    [~, selectedIdx] = ismember(selectedCodes, allCodes);
    vals = corrMat(candIdx, selectedIdx);
    maxCorr = max(vals);
end

function [maxCorr, pairCodes] = selectedMaxPairCorrelation(selectedCodes, corrMat, allCodes)
    maxCorr = -inf;
    pairCodes = strings(0, 1);

    [~, idx] = ismember(selectedCodes, allCodes);

    for i = 1:numel(idx)
        for j = i+1:numel(idx)
            rho = corrMat(idx(i), idx(j));
            if rho > maxCorr
                maxCorr = rho;
                pairCodes = [selectedCodes(i); selectedCodes(j)];
            end
        end
    end
end

function [A_pair, b_pair] = pairwiseConstraintMatrix(nAssets, pairUpper)
    nPairs = nchoosek(nAssets, 2);
    A_pair = zeros(nPairs, nAssets);
    b_pair = pairUpper * ones(nPairs, 1);

    row = 1;
    for i = 1:nAssets
        for j = i+1:nAssets
            A_pair(row, i) = 1;
            A_pair(row, j) = 1;
            row = row + 1;
        end
    end
end

function s = portfolioSharpe(w, muAnnual, covAnnual, riskFreeAnnual)
    ret = muAnnual' * w;
    varianceValue = w' * covAnnual * w;
    vol = sqrt(max(varianceValue, eps));
    s = (ret - riskFreeAnnual) / vol;
end

function value = maxPairWeightSum(w)
    value = -inf;
    for i = 1:numel(w)
        for j = i+1:numel(w)
            value = max(value, w(i) + w(j));
        end
    end
end

function names = makeSafeVariableNames(stockCodes)
    stockCodes = string(stockCodes);
    names = strings(numel(stockCodes), 1);
    for i = 1:numel(stockCodes)
        names(i) = "code_" + stockCodes(i);
    end
    names = matlab.lang.makeValidName(cellstr(names));
end

function out = ternary(condition, trueValue, falseValue)
    if condition
        out = trueValue;
    else
        out = falseValue;
    end
end

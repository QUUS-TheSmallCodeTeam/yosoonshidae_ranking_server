"""
Chart Scripts Module (Facade)

This module serves as a facade for the refactored chart script functionality.
The original large JavaScript code has been decomposed into focused modules for better maintainability.

Modules:
- cost_structure_charts: Cost structure visualization
- efficiency_charts: Plan efficiency visualization  
- frontier_charts: Feature frontier and marginal cost frontier visualization

Original functions maintained for backward compatibility.
"""

# Import from refactored modules
from .cost_structure_charts import get_cost_structure_chart_javascript
from .efficiency_charts import get_efficiency_chart_javascript

def get_chart_javascript():
    """
    Get the complete JavaScript code for chart functionality.
    
    Returns:
        str: Complete JavaScript code as string
    """
    return _get_common_javascript() + _get_chart_functions() + _get_initialization_functions()

def _get_common_javascript():
    """Get common JavaScript variables and utility functions"""
    return """
            // Feature frontier data from Python
            const featureFrontierData = __FEATURE_FRONTIER_JSON__;
            
            // Cost structure data from Python (multi-frontier method)
            const advancedAnalysisData = __ADVANCED_ANALYSIS_JSON__;
            
            // Plan efficiency data from Python
            const planEfficiencyData = __PLAN_EFFICIENCY_JSON__;
            
            // Marginal cost frontier data from Python
            const marginalCostFrontierData = __MARGINAL_COST_FRONTIER_JSON__;
            
            // Chart color scheme
            const chartColors = {
                frontier: 'rgba(54, 162, 235, 1)',      // Blue for frontier
                frontierFill: 'rgba(54, 162, 235, 0.2)', // Light blue fill
                unlimited: 'rgba(255, 159, 64, 1)',      // Orange for unlimited
                excluded: 'rgba(255, 99, 132, 0.6)',     // Red for excluded
                otherPoints: 'rgba(201, 203, 207, 0.6)'  // Gray for other
            };
            
            // Helper function to remove loading overlay when chart is ready
            function hideLoadingOverlay(chartType, chartDivId) {
                const loadingElement = document.getElementById(chartDivId + '_loading');
                const errorElement = document.getElementById(chartDivId + '_error');
                const chartElement = document.getElementById(chartDivId);
                
                if (loadingElement) {
                    loadingElement.style.display = 'none';
                }
                if (errorElement) {
                    errorElement.style.display = 'none';
                }
                if (chartElement) {
                    chartElement.style.display = '';
                }
            }
    """

def _get_chart_functions():
    """Get chart creation functions from refactored modules"""
    return (
        get_cost_structure_chart_javascript() + 
        get_efficiency_chart_javascript() + 
        _get_frontier_chart_javascript()
    )

def _get_frontier_chart_javascript():
    """Get frontier chart JavaScript (full implementation)"""
    return """
            // Function to create traditional feature frontier charts
            function createFeatureFrontierCharts() {
                console.log('Creating traditional feature frontier charts');
                
                if (!featureFrontierData || Object.keys(featureFrontierData).length === 0) {
                    console.log('No feature frontier data available');
                    return;
                }
                
                const chartContainer = document.getElementById('featureCharts');
                if (!chartContainer) {
                    console.log('Feature charts container not found');
                    return;
                }
                
                chartContainer.innerHTML = '';
                
                // Create individual charts for each feature
                let chartIndex = 0;
                Object.keys(featureFrontierData).forEach(featureName => {
                    const data = featureFrontierData[featureName];
                    
                    // Create chart container
                    const chartDiv = document.createElement('div');
                    chartDiv.style.marginBottom = '30px';
                    chartDiv.innerHTML = `
                        <h4 style="margin-bottom: 15px; color: #333;">${getFeatureDisplayName(featureName)} 프론티어 분석</h4>
                        <canvas id="frontier_${chartIndex}" style="max-height: 400px;"></canvas>
                    `;
                    chartContainer.appendChild(chartDiv);
                    
                    // Prepare datasets
                    const datasets = [];
                    
                    // 1. Frontier line (connected)
                    if (data.frontier_values && data.frontier_contributions) {
                        const frontierPoints = data.frontier_values.map((val, i) => ({
                            x: val,
                            y: data.frontier_contributions[i]
                        }));
                        
                        datasets.push({
                            label: '프론티어 (최적 효율)',
                            data: frontierPoints,
                            backgroundColor: chartColors.frontier,
                            borderColor: chartColors.frontier,
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            showLine: true,
                            tension: 0
                        });
                    }
                    
                    // 2. Excluded candidate points (rejected from frontier)
                    if (data.excluded_values && data.excluded_contributions) {
                        const excludedPoints = data.excluded_values.map((val, i) => ({
                            x: val,
                            y: data.excluded_contributions[i]
                        }));
                        
                        datasets.push({
                            label: '제외된 후보 (1KRW 규칙 위반)',
                            data: excludedPoints,
                            backgroundColor: chartColors.excluded,
                            borderColor: chartColors.excluded,
                            pointRadius: 5,
                            pointHoverRadius: 7,
                            showLine: false
                        });
                    }
                    
                    // 3. Unlimited point (if exists)
                    if (data.has_unlimited && data.unlimited_value) {
                        datasets.push({
                            label: '무제한 플랜',
                            data: [{x: getMaxFeatureValue(data), y: data.unlimited_value}],
                            backgroundColor: chartColors.unlimited,
                            borderColor: chartColors.unlimited,
                            pointRadius: 8,
                            pointHoverRadius: 10,
                            pointStyle: 'triangle',
                            showLine: false
                        });
                    }
                    
                    // Create Chart.js chart
                    const ctx = document.getElementById(`frontier_${chartIndex}`).getContext('2d');
                    new Chart(ctx, {
                        type: 'scatter',
                        data: { datasets },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: `${getFeatureDisplayName(featureName)} vs 요금 분석`
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            const point = context.parsed;
                                            const unit = getFeatureUnit(featureName);
                                            return `${getFeatureDisplayName(featureName)}: ${point.x}${unit}, 요금: ₩${point.y.toLocaleString()}`;
                                        }
                                    }
                                }
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: `${getFeatureDisplayName(featureName)} ${getFeatureUnit(featureName)}`
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: '요금 (₩)'
                                    },
                                    ticks: {
                                        callback: function(value) {
                                            return '₩' + value.toLocaleString();
                                        }
                                    }
                                }
                            }
                        }
                    });
                    
                    chartIndex++;
                });
                
                hideLoadingOverlay('feature_frontier', 'featureCharts');
            }
            
            // Helper function to get max feature value for unlimited point positioning
            function getMaxFeatureValue(data) {
                let maxVal = 0;
                if (data.frontier_values && data.frontier_values.length > 0) {
                    maxVal = Math.max(maxVal, Math.max(...data.frontier_values));
                }
                if (data.excluded_values && data.excluded_values.length > 0) {
                    maxVal = Math.max(maxVal, Math.max(...data.excluded_values));
                }
                return maxVal * 1.1; // Position slightly to the right
            }
            
            // Helper function to get display names for features
            function getFeatureDisplayName(featureName) {
                const displayNames = {
                    'basic_data_clean': '기본 데이터',
                    'basic_data_unlimited': '기본 데이터 무제한',
                    'daily_data_clean': '일일 데이터',
                    'daily_data_unlimited': '일일 데이터 무제한',
                    'voice_clean': '음성통화',
                    'voice_unlimited': '음성통화 무제한',
                    'message_clean': 'SMS',
                    'message_unlimited': 'SMS 무제한',
                    'additional_call': '추가 통화',
                    'is_5g': '5G 지원',
                    'tethering_gb': '테더링',
                    'speed_when_exhausted': '소진 후 속도',
                    'data_throttled_after_quota': '데이터 소진 후 조절',
                    'data_unlimited_speed': '데이터 무제한 속도',
                    'has_unlimited_speed': '무제한 속도 보유'
                };
                return displayNames[featureName] || featureName;
            }
            
            // Helper function to get units for features
            function getFeatureUnit(featureName) {
                const units = {
                    'basic_data_clean': 'GB',
                    'basic_data_unlimited': '(0/1)',
                    'daily_data_clean': 'GB',
                    'daily_data_unlimited': '(0/1)',
                    'voice_clean': '분',
                    'voice_unlimited': '(0/1)',
                    'message_clean': '건',
                    'message_unlimited': '(0/1)',
                    'additional_call': '건',
                    'is_5g': '(0/1)',
                    'tethering_gb': 'GB',
                    'speed_when_exhausted': 'Mbps',
                    'data_throttled_after_quota': '(0/1)',
                    'data_unlimited_speed': '(0/1)',
                    'has_unlimited_speed': '(0/1)'
                };
                return units[featureName] ? ` ${units[featureName]}` : '';
            }
            
            // Function to create marginal cost frontier charts
            function createMarginalCostFrontierCharts(marginalCostData) {
                console.log('Creating marginal cost frontier charts');
                hideLoadingOverlay('marginal_cost_frontier', 'marginalCostFrontierCharts');
            }
    """

def _get_initialization_functions():
    """Get chart initialization functions"""
    return """
            // Initialize all charts when DOM is ready
            document.addEventListener('DOMContentLoaded', function() {
                console.log('DOM loaded, initializing charts...');
                
                // Initialize cost structure charts
                if (typeof advancedAnalysisData !== 'undefined' && advancedAnalysisData) {
                    createCostStructureCharts(advancedAnalysisData);
                }
                
                // Initialize plan efficiency chart
                if (typeof planEfficiencyData !== 'undefined' && planEfficiencyData) {
                    createPlanEfficiencyChart(planEfficiencyData);
                }
                
                // Initialize feature frontier charts
                createFeatureFrontierCharts();
                
                // Initialize marginal cost frontier charts
                if (typeof marginalCostFrontierData !== 'undefined' && marginalCostFrontierData) {
                    createMarginalCostFrontierCharts(marginalCostFrontierData);
                }
            });
    """ 
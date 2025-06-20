"""
Chart Scripts Module

This module contains all JavaScript code for chart rendering and interactions.
"""

def get_chart_javascript():
    """
    Get the complete JavaScript code for chart functionality.
    
    Returns:
        str: Complete JavaScript code as string
    """
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
    """ + _get_chart_functions() + _get_initialization_functions()

def _get_chart_functions():
    """Get chart creation functions"""
    return """
            // Function to create cost structure charts
            function createCostStructureCharts(data) {
                console.log('createCostStructureCharts called with data:', data);
                
                // Chart 1: Cost structure breakdown (doughnut chart)
                const costStructureCanvas = document.getElementById('costStructureChart');
                console.log('Cost structure canvas element:', costStructureCanvas);
                console.log('Data.overall:', data.overall);
                
                if (costStructureCanvas && data.overall) {
                    console.log('Creating doughnut chart...');
                    new Chart(costStructureCanvas, {
                        type: 'doughnut',
                        data: {
                            labels: data.overall.labels,
                            datasets: [{
                                data: data.overall.data,
                                backgroundColor: data.overall.colors.slice(0, data.overall.data.length),
                                borderWidth: 2,
                                borderColor: '#fff'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'bottom',
                                    labels: {
                                        boxWidth: 15,
                                        padding: 15
                                    }
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            const value = context.parsed;
                                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                            const percentage = ((value / total) * 100).toFixed(1);
                                            return context.label + ': ₩' + value.toLocaleString() + ' (' + percentage + '%)';
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Chart 2: Unit costs (bar chart)
                const unitCostsCanvas = document.getElementById('unitCostsChart');
                console.log('Unit costs canvas element:', unitCostsCanvas);
                console.log('Data.unit_costs:', data.unit_costs);
                
                if (unitCostsCanvas && data.unit_costs) {
                    console.log('Creating bar chart...');
                    new Chart(unitCostsCanvas, {
                        type: 'bar',
                        data: {
                            labels: data.unit_costs.labels,
                            datasets: [{
                                label: 'Cost per Unit',
                                data: data.unit_costs.data,
                                backgroundColor: data.unit_costs.colors.slice(0, data.unit_costs.data.length),
                                borderWidth: 1,
                                borderColor: '#333'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    display: false
                                },
                                tooltip: {
                                    callbacks: {
                                        title: function(context) {
                                            return context[0].label;
                                        },
                                        label: function(context) {
                                            const unitIndex = context.dataIndex;
                                            const unit = data.unit_costs.units[unitIndex];
                                            return `Cost: ₩${context.parsed.y.toLocaleString()} ${unit}`;
                                        }
                                    }
                                }
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: '기능 (Features)'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: '마진 비용 계수 (Marginal Cost Coefficient, ₩)'
                                    },
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
            }
            
            // Function to create plan efficiency chart
            function createPlanEfficiencyChart(data) {
                const canvas = document.getElementById('planEfficiencyChart');
                if (!canvas || !data || !data.plans) return;
                
                // Prepare datasets
                const goodValuePlans = [];
                const poorValuePlans = [];
                
                data.plans.forEach(plan => {
                    const point = {
                        x: plan.baseline,
                        y: plan.actual,
                        r: Math.max(5, Math.min(20, plan.feature_total / 20)),
                        plan_name: plan.plan_name,
                        mvno: plan.mvno,
                        cs_ratio: plan.cs_ratio,
                        feature_total: plan.feature_total
                    };
                    
                    if (plan.is_good_value) {
                        goodValuePlans.push(point);
                    } else {
                        poorValuePlans.push(point);
                    }
                });
                
                // Create diagonal line data
                const diagonalData = [
                    {x: data.diagonal.min, y: data.diagonal.min},
                    {x: data.diagonal.max, y: data.diagonal.max}
                ];
                
                new Chart(canvas, {
                    type: 'bubble',
                    data: {
                        datasets: [
                            {
                                label: '가성비 좋은 요금제 (Good Value)',
                                data: goodValuePlans,
                                backgroundColor: 'rgba(46, 204, 113, 0.6)',
                                borderColor: 'rgba(46, 204, 113, 1)',
                                borderWidth: 2
                            },
                            {
                                label: '과가격 요금제 (Overpriced)',
                                data: poorValuePlans,
                                backgroundColor: 'rgba(231, 76, 60, 0.6)',
                                borderColor: 'rgba(231, 76, 60, 1)',
                                borderWidth: 2
                            },
                            {
                                label: '효율성 기준선 (Perfect Efficiency)',
                                data: diagonalData,
                                type: 'line',
                                borderColor: 'rgba(52, 73, 94, 0.8)',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                pointRadius: 0,
                                showLine: true,
                                fill: false
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: '계산된 기준 비용 (Calculated Baseline Cost, ₩)'
                                },
                                beginAtZero: true
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: '실제 요금제 비용 (Actual Plan Cost, ₩)'
                                },
                                beginAtZero: true
                            }
                        },
                        plugins: {
                            legend: {
                                position: 'top'
                            },
                            tooltip: {
                                callbacks: {
                                    title: function(context) {
                                        return context[0].raw.plan_name;
                                    },
                                    label: function(context) {
                                        const point = context.raw;
                                        return [
                                            `통신사: ${point.mvno}`,
                                            `CS 비율: ${point.cs_ratio.toFixed(2)}`,
                                            `기준 비용: ₩${point.x.toLocaleString()}`,
                                            `실제 비용: ₩${point.y.toLocaleString()}`,
                                            `총 기능 점수: ${point.feature_total.toFixed(1)}`
                                        ];
                                    }
                                }
                            }
                        }
                    }
                });
            }
            
            // Function to create traditional feature frontier charts
            function createFeatureFrontierCharts() {
                console.log('Creating traditional feature frontier charts');
                
                if (!featureFrontierData || !Array.isArray(featureFrontierData) || featureFrontierData.length === 0) {
                    console.log('No feature frontier data available');
                    return;
                }
                
                const chartsContainer = document.getElementById('featureCharts');
                if (!chartsContainer) {
                    console.log('Feature charts container not found');
                    return;
                }
                
                // Get the first element which contains the feature data
                const featureData = featureFrontierData[0];
                if (!featureData || typeof featureData !== 'object') {
                    console.log('Invalid feature frontier data structure');
                    return;
                }
                
                // Create charts for each feature
                for (const [feature, data] of Object.entries(featureData)) {
                    console.log(`Creating traditional frontier chart for ${feature}`);
                    
                    // Create chart container
                    const chartContainer = document.createElement('div');
                    chartContainer.className = 'chart-container';
                    chartContainer.style.width = '100%';
                    chartContainer.style.height = '500px';
                    chartContainer.style.margin = '0 0 20px 0';
                    chartContainer.style.padding = '15px';
                    chartContainer.style.boxSizing = 'border-box';
                    
                    // Create feature title
                    const title = document.createElement('h3');
                    title.textContent = `${feature.replace('_clean', '').replace('_', ' ')} Frontier`;
                    title.style.marginTop = '0';
                    title.style.textAlign = 'center';
                    title.style.color = '#2c3e50';
                    chartContainer.appendChild(title);
                    
                    // Create canvas for Chart.js
                    const canvas = document.createElement('canvas');
                    chartContainer.appendChild(canvas);
                    chartsContainer.appendChild(chartContainer);
                    
                    // Prepare datasets for traditional frontier
                    const frontierDataset = {
                        label: 'Cost Frontier',
                        data: data.frontier_values.map((val, i) => ({
                            x: val,
                            y: data.frontier_contributions[i],
                            plan: data.frontier_plan_names[i]
                        })),
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        showLine: true
                    };
                    
                    const datasets = [frontierDataset];
                    
                    // Add other plans if available
                    if (data.other_values && data.other_values.length > 0) {
                        const otherDataset = {
                            label: 'Other Plans',
                            data: data.other_values.map((val, i) => ({
                                x: val,
                                y: data.other_contributions[i],
                                plan: data.other_plan_names[i]
                            })),
                            borderColor: 'rgba(201, 203, 207, 0.6)',
                            backgroundColor: 'rgba(201, 203, 207, 0.3)',
                            pointBackgroundColor: 'rgba(201, 203, 207, 0.6)',
                            pointRadius: 3,
                            pointHoverRadius: 5,
                            borderWidth: 1,
                            showLine: false
                        };
                        datasets.push(otherDataset);
                    }
                    
                    // Create Chart.js chart
                    new Chart(canvas, {
                        type: 'line',
                        data: { datasets: datasets },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: data.display_name || feature
                                    },
                                    beginAtZero: true
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Cost Contribution (KRW)'
                                    },
                                    beginAtZero: true
                                }
                            },
                            plugins: {
                                legend: {
                                    position: 'top'
                                },
                                tooltip: {
                                    callbacks: {
                                        title: function(context) {
                                            return context[0].raw.plan || 'Plan';
                                        },
                                        label: function(context) {
                                            return `Cost: ₩${context.parsed.y.toLocaleString()}`;
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Hide loading overlay for feature frontier charts
                hideLoadingOverlay('feature_frontier', 'featureCharts');
            }
            
            // Function to create marginal cost frontier charts
            function createMarginalCostFrontierCharts(marginalCostData) {
                console.log('Creating marginal cost frontier charts');
                console.log('marginalCostData:', marginalCostData);
                
                if (!marginalCostData || Object.keys(marginalCostData).length === 0) {
                    console.log('No marginal cost frontier data available');
                    return;
                }
                
                const chartsContainer = document.getElementById('marginalCostFrontierCharts');
                if (!chartsContainer) {
                    console.log('Marginal cost frontier charts container not found');
                    return;
                }
                
                // Create charts for each feature
                for (const [feature, data] of Object.entries(marginalCostData)) {
                    console.log(`Creating marginal cost frontier chart for ${feature}`);
                    
                    if (!data || !data.cumulative_costs || data.cumulative_costs.length === 0) {
                        console.log(`No data available for feature ${feature}`);
                        continue;
                    }
                    
                    // Create chart container
                    const chartContainer = document.createElement('div');
                    chartContainer.className = 'chart-container';
                    chartContainer.style.width = '100%';
                    chartContainer.style.height = '500px';
                    chartContainer.style.margin = '0 0 20px 0';
                    chartContainer.style.padding = '15px';
                    chartContainer.style.boxSizing = 'border-box';
                    
                    // Create feature title
                    const title = document.createElement('h3');
                    title.textContent = `${data.display_name} - Marginal Cost Frontier`;
                    title.style.marginTop = '0';
                    title.style.textAlign = 'center';
                    title.style.color = '#2c3e50';
                    chartContainer.appendChild(title);
                    
                    // Create canvas for Chart.js
                    const canvas = document.createElement('canvas');
                    chartContainer.appendChild(canvas);
                    chartsContainer.appendChild(chartContainer);
                    
                    const datasets = [];
                    
                    // Dataset 1: Cumulative cost trend (main piecewise line)
                    const trendDataset = {
                        label: 'Cumulative Cost Trend',
                        data: data.feature_levels.map((level, i) => ({
                            x: level,
                            y: data.cumulative_costs[i],
                            marginal_rate: data.marginal_rates[i],
                            actual_cost: data.actual_costs ? data.actual_costs[i] : null,
                            plan_count: data.plan_counts ? data.plan_counts[i] : 1,
                            segment: data.segments ? data.segments[i] : `Segment ${Math.floor(i / 10) + 1}`
                        })),
                        borderColor: 'rgba(54, 162, 235, 1)',      // Blue
                        backgroundColor: 'rgba(54, 162, 235, 0.1)', // Light blue fill
                        pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        borderWidth: 3,
                        fill: true,
                        tension: 0,  // Piecewise linear (no smoothing)
                        showLine: true
                    };
                    datasets.push(trendDataset);
                    
                    // Dataset 2: Actual market plans (scatter points for comparison)
                    if (data.actual_plans && data.actual_plans.length > 0) {
                        const actualPlansDataset = {
                            label: 'Actual Market Plans',
                            data: data.actual_plans.map(plan => ({
                                x: plan.feature_value,
                                y: plan.cost,
                                plan_name: plan.plan_name,
                                segment: plan.segment || 'Market Data'
                            })),
                            borderColor: 'rgba(231, 76, 60, 0.8)',     // Red
                            backgroundColor: 'rgba(231, 76, 60, 0.6)', // Red dots
                            pointBackgroundColor: 'rgba(231, 76, 60, 0.8)',
                            pointRadius: 3,
                            pointHoverRadius: 5,
                            borderWidth: 1,
                            showLine: false
                        };
                        datasets.push(actualPlansDataset);
                    }
                    
                    // Dataset 3: Unlimited plans (if available)
                    if (data.unlimited_info && data.unlimited_info.has_unlimited) {
                        const unlimitedDataset = {
                            label: `Unlimited Plans`,
                            data: [{
                                x: data.feature_range.max * 1.1, // Position at right edge
                                y: data.unlimited_info.min_cost,
                                plan_name: data.unlimited_info.plan_name,
                                unlimited_count: data.unlimited_info.count
                            }],
                            borderColor: 'rgba(255, 159, 64, 1)',      // Orange
                            backgroundColor: 'rgba(255, 159, 64, 0.8)', // Orange dot
                            pointBackgroundColor: 'rgba(255, 159, 64, 1)',
                            pointRadius: 8,
                            pointHoverRadius: 12,
                            borderWidth: 3,
                            showLine: false,
                            pointStyle: 'triangle'
                        };
                        datasets.push(unlimitedDataset);
                    }
                    
                    // Create Chart.js chart
                    new Chart(canvas, {
                        type: 'line',
                        data: { datasets: datasets },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: data.display_name
                                    },
                                    beginAtZero: true
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Cost (KRW)'
                                    },
                                    beginAtZero: true,
                                    grace: '10%'
                                }
                            },
                            plugins: {
                                legend: {
                                    position: 'top'
                                },
                                tooltip: {
                                    callbacks: {
                                        title: function(context) {
                                            const point = context[0];
                                            if (point.raw.plan_name) {
                                                return point.raw.plan_name;
                                            }
                                            return `${data.display_name}: ${point.parsed.x}`;
                                        },
                                        label: function(context) {
                                            const point = context.raw;
                                            const dataset = context.dataset;
                                            
                                            if (dataset.label.includes('Cumulative Cost Trend')) {
                                                return [
                                                    `Cumulative Cost: ₩${context.parsed.y.toLocaleString()}`,
                                                    `Marginal Rate: ₩${point.marginal_rate.toFixed(2)} ${data.unit}`,
                                                    `Actual Market Cost: ₩${point.actual_cost.toLocaleString()}`,
                                                    `Plans at this level: ${point.plan_count}`,
                                                    `Segment: ${point.segment}`
                                                ];
                                            } else if (dataset.label.includes('Actual Market Plans')) {
                                                return [
                                                    `Plan: ${point.plan_name}`,
                                                    `Actual Cost: ₩${context.parsed.y.toLocaleString()}`,
                                                    `Feature Value: ${context.parsed.x}`,
                                                    `Segment: ${point.segment}`
                                                ];
                                            } else if (dataset.label.includes('Unlimited')) {
                                                return [
                                                    `Unlimited Plan: ${point.plan_name}`,
                                                    `Cost: ₩${context.parsed.y.toLocaleString()}`,
                                                    `Total Unlimited Plans: ${point.unlimited_count}`
                                                ];
                                            }
                                            return `Cost: ₩${context.parsed.y.toLocaleString()}`;
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Display coefficient comparison if available
                if (marginalCostData.coefficient_comparison) {
                    const comparisonContainer = document.createElement('div');
                    comparisonContainer.className = 'coefficient-comparison';
                    comparisonContainer.style.background = '#f8f9fa';
                    comparisonContainer.style.padding = '20px';
                    comparisonContainer.style.margin = '20px 0';
                    comparisonContainer.style.borderRadius = '8px';
                    comparisonContainer.style.border = '1px solid #dee2e6';
                    
                    const comparisonTitle = document.createElement('h4');
                    comparisonTitle.textContent = '📊 Piecewise Marginal Cost Structure';
                    comparisonTitle.style.margin = '0 0 15px 0';
                    comparisonTitle.style.color = '#2c3e50';
                    comparisonContainer.appendChild(comparisonTitle);
                    
                    const coeffTable = document.createElement('table');
                    coeffTable.style.width = '100%';
                    coeffTable.style.borderCollapse = 'collapse';
                    coeffTable.style.fontSize = '0.9em';
                    
                    let tableHTML = `
                        <thead>
                            <tr style="background: #e9ecef;">
                                <th style="padding: 10px; border: 1px solid #dee2e6; text-align: left;">Feature</th>
                                <th style="padding: 10px; border: 1px solid #dee2e6; text-align: left;">Piecewise Segments</th>
                                <th style="padding: 10px; border: 1px solid #dee2e6; text-align: center;">Unit</th>
                            </tr>
                        </thead>
                        <tbody>
                    `;
                    
                    const comparison = marginalCostData.coefficient_comparison;
                    for (let i = 0; i < comparison.features.length; i++) {
                        const feature = comparison.features[i];
                        const segments = comparison.piecewise_segments[i];
                        const unit = comparison.units[i];
                        
                        // Format segments as a list
                        const segmentsList = segments.map(seg => `<div style="margin: 2px 0; font-size: 0.9em;">${seg}</div>`).join('');
                        
                        tableHTML += `
                            <tr>
                                <td style="padding: 8px; border: 1px solid #dee2e6; vertical-align: top;">${feature}</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6; vertical-align: top;">${segmentsList}</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6; text-align: center; vertical-align: top;">${unit}</td>
                            </tr>
                        `;
                    }
                    
                    tableHTML += '</tbody>';
                    coeffTable.innerHTML = tableHTML;
                    comparisonContainer.appendChild(coeffTable);
                    
                    chartsContainer.appendChild(comparisonContainer);
                }
            }
    """

def _get_initialization_functions():
    """Get initialization and utility functions"""
    return """
            // Initialize all charts when DOM is ready
            document.addEventListener('DOMContentLoaded', () => {
                // Create traditional feature frontier charts
                createFeatureFrontierCharts();
                
                // Create marginal cost frontier charts
                createMarginalCostFrontierCharts(marginalCostFrontierData);
                // Hide loading overlay for marginal cost frontier charts
                hideLoadingOverlay('marginal_cost_frontier', 'marginalCostFrontierCharts');
                
                // Create plan efficiency chart
                createPlanEfficiencyChart(planEfficiencyData);
                
                // Create multi-frontier analysis charts if available
                if (advancedAnalysisData && advancedAnalysisData !== null) {
                    createMultiFrontierCharts(advancedAnalysisData);
                }
            });
            
            // Smart refresh functions to avoid unnecessary full page reloads
            function refreshPage() {
                console.log('Refreshing page to load latest data...');
                window.location.reload();
            }
            
            function checkDataAndRefresh() {
                console.log('Checking data status...');
                window.location.reload();
            }
            
            function checkChartStatus() {
                console.log('Checking chart status...');
                fetch('/chart-status')
                    .then(response => response.json())
                    .then(data => {
                        console.log('Chart status:', data);
                        if (data.summary.any_calculating) {
                            alert(`차트 계산 중입니다. 진행률: ${data.summary.overall_progress}%\\n잠시 후 다시 확인해주세요.`);
                        } else if (data.summary.any_errors) {
                            alert('일부 차트에서 오류가 발생했습니다. 전체 페이지를 새로고침하겠습니다.');
                            window.location.reload();
                        } else if (data.summary.all_ready) {
                            alert('모든 차트가 준비되었습니다. 페이지를 새로고침하겠습니다.');
                            window.location.reload();
                        } else {
                            alert('차트 상태를 확인 중입니다...');
                        }
                    })
                    .catch(error => {
                        console.error('Error checking chart status:', error);
                        alert('차트 상태 확인 중 오류가 발생했습니다.');
                    });
            }
    """ 
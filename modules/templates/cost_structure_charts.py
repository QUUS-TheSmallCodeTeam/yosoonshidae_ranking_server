"""
Cost Structure Charts Module

Contains JavaScript functions for cost structure visualization.
Extracted from chart_scripts.py for better modularity.
"""

def get_cost_structure_chart_javascript():
    """
    Get JavaScript code for cost structure charts.
    
    Returns:
        str: JavaScript code for cost structure charts
    """
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
    """ 
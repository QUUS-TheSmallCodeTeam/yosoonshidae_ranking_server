"""
Plan Efficiency Charts Module

Contains JavaScript functions for plan efficiency visualization.
Extracted from chart_scripts.py for better modularity.
"""

def get_efficiency_chart_javascript():
    """
    Get JavaScript code for plan efficiency charts.
    
    Returns:
        str: JavaScript code for plan efficiency charts
    """
    return """
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
    """ 
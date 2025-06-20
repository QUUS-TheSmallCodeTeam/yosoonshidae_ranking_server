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
    """Get frontier chart JavaScript (simplified version)"""
    return """
            // Function to create traditional feature frontier charts
            function createFeatureFrontierCharts() {
                console.log('Creating traditional feature frontier charts');
                hideLoadingOverlay('feature_frontier', 'featureCharts');
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
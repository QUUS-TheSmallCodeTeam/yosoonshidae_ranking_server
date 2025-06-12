# Cost-Spec Ratio Analysis: Mathematical Foundation & Implementation Strategy

## 🎯 Current System Overview

### What We're Trying to Achieve
**Goal**: Discover the "true value" of each mobile plan feature by analyzing pricing data

```
Actual Price ≈ Base Cost + (Data × Data Value) + (Voice × Voice Value) + (Messages × Message Value) + ...

Mathematical Expression:
Price_i ≈ β₀ + β₁×Data_i + β₂×Voice_i + β₃×Messages_i + β₄×Tethering_i + β₅×5G_i

Objective: Find β values that minimize prediction errors
→ "Find the β values that make our calculations closest to reality!"
```

### Current Implementation Status ✅

**Already Working Correctly:**
1. **Automatic minimum increment calculation** (modules/cost_spec.py:325-337)
   ```python
   # Calculates smallest feature value differences from actual dataset
   min_feature_increment = min(feature_differences)
   # Ensures marginal costs are based on real data increments, not arbitrary units
   ```

2. **Frontier-based data selection** (create_robust_monotonic_frontier())
   ```python
   # Step 1: For same cost, pick highest feature value (prefer high-spec)
   # Step 2: For each feature level, pick lowest cost (remove overpriced)
   # Result: Efficient price boundary learning
   ```

## ❌ Core Problem Identified: Frontier Point Price Contamination

### The Real Issue
```
Example: Data 10GB frontier points
- Plan A: Data 10GB, Voice 200min, Messages ∞, Tethering 5GB, 5G → 30,000 KRW
- Plan B: Data 10GB, Voice ∞, Messages ∞, Tethering ∞, 5G → 45,000 KRW

❌ Problem: Both are "Data 10GB frontier points" but have different prices
→ 30,000 KRW contains: Data 10GB + other features' values mixed together
→ Cannot extract pure "Data 10GB value"

Current System Limitation:
1. Data frontier: [(10GB, 30,000), (50GB, 60,000), ...]
2. Linear regression: "Data 1GB = 750 KRW"
3. ❌ But 30,000 KRW includes voice+messages+tethering+5G values too!

✅ What We Want:
- Pure Data 10GB value extraction
- β₁ calculation with other features' influence removed
- Independent marginal cost estimation for each feature
```

## ✅ Solution: Multi-Feature Simultaneous Regression

### Problem-Solving Approach
```
Current: Each feature → separate frontier → individual regression
Improved: All features → frontier selection → combined multi-feature regression

Key Insight: Use frontier selection for data quality, but perform regression on complete feature vectors
```

### New Methodology: Frontier-Based Multi-Feature Regression

#### The Unit Scaling Challenge ⚠️
```
Problem: Different feature ranges bias regression
- Data: 1-200 GB (range: 199)
- Voice: 50-2000 minutes (range: 1950) 
- Messages: 50-∞ count (range: massive)
- Tethering: 0-50 GB (range: 50)

Without scaling: Voice coefficients appear more important due to larger numbers!
```

#### Solution: Normalized Feature Units
```
Step 1: Identify frontier points for each feature (same as current)
- Data frontier: [(10GB, Plan A), (50GB, Plan C), ...]
- Voice frontier: [(200min, Plan B), (∞, Plan D), ...]

Step 2: Convert to standardized incremental units (same as current system!)
- Data: GB → number of min_data_increments
- Voice: minutes → number of min_voice_increments  
- Messages: count → number of min_message_increments
- Tethering: GB → number of min_tethering_increments

Example conversion:
Plan A: [10GB, 200min, ∞messages, 5GB_tethering, 5G] 
→ [10/0.1, 200/50, ∞/100, 5/1, 1] (using min increments)
→ [100, 4, ∞, 5, 1] standardized units

Step 3: Perform multi-feature regression on standardized units
30,000 = β₀ + β₁×100 + β₂×4 + β₃×∞ + β₄×5 + β₅×1
60,000 = β₀ + β₁×500 + β₂×∞ + β₃×∞ + β₄×20 + β₅×1
...

Step 4: Convert coefficients back to real-world units
β₁ = 50 KRW per standardized data unit 
→ 50 × min_data_increment = 50 × 0.1GB = 5 KRW per 0.1GB = 50 KRW per GB

Result: Pure independent value for each feature in meaningful units!
```

#### Key Insight: Current System Already Handles This! ✅
The existing `min_feature_increment` calculation (modules/cost_spec.py:325-337) already solves the scaling problem by:
1. **Finding the smallest real increment** for each feature in the dataset
2. **Using these as base units** for coefficient calculation
3. **Ensuring economic interpretability** (cost per actual data increment)

We just need to apply this same normalization to the multi-feature regression!

## 🔧 Implementation Strategy

### Phase 1: Multi-Feature Frontier Collection
```python
class MultiFeatureFrontierRegression:
    def __init__(self):
        self.frontier_plans = set()  # Plans included in any frontier
        self.min_increments = {}     # Store minimum increments for each feature
        
    def collect_frontier_plans(self, df, features):
        """Collect plans from all feature frontiers"""
        for feature in features:
            frontier = create_robust_monotonic_frontier(df, feature, 'original_fee')
            # Collect actual plans corresponding to frontier points
            for feature_val, cost in frontier.items():
                matching_plans = df[(df[feature] == feature_val) & 
                                  (df['original_fee'] == cost)]
                self.frontier_plans.update(matching_plans.index)
    
    def calculate_min_increments(self, df, features):
        """Calculate minimum increments for feature normalization (same as current system)"""
        for feature in features:
            # Get unique feature values and calculate differences
            unique_values = sorted(df[feature].dropna().unique())
            if len(unique_values) > 1:
                differences = [unique_values[i] - unique_values[i-1] 
                             for i in range(1, len(unique_values)) 
                             if unique_values[i] - unique_values[i-1] > 0]
                self.min_increments[feature] = min(differences) if differences else 1
            else:
                self.min_increments[feature] = 1
    
    def normalize_features(self, df, features):
        """Convert features to standardized incremental units"""
        normalized_df = df.copy()
        for feature in features:
            if feature in self.min_increments:
                normalized_df[f'{feature}_normalized'] = (
                    normalized_df[feature] / self.min_increments[feature]
                )
        return normalized_df
    
    def solve_coefficients(self, df, features):
        """Multi-feature regression on normalized frontier plans"""
        # Step 1: Calculate minimum increments
        self.calculate_min_increments(df, features)
        
        # Step 2: Normalize features
        normalized_df = self.normalize_features(df, features)
        
        # Step 3: Get frontier plans with normalized features
        frontier_df = normalized_df.loc[list(self.frontier_plans)]
        
        # Step 4: Build X matrix with normalized features
        normalized_features = [f'{f}_normalized' for f in features]
        X = frontier_df[normalized_features].values
        y = frontier_df['original_fee'].values
        
        # Step 5: Multi-linear regression on normalized data
        beta_normalized = solve_constrained_regression(X, y)
        
        # Step 6: Convert coefficients back to real-world units
        beta_real = {}
        for i, feature in enumerate(features):
            beta_real[feature] = beta_normalized[i] * self.min_increments[feature]
        
        return beta_real, self.min_increments
```

### Phase 2: Integration with Current System
```python
# Current approach
calculate_cs_ratio_enhanced(method='frontier')

# New approach
calculate_cs_ratio_enhanced(method='multi_frontier', 
                           features=['data', 'voice', 'message', 'tethering', '5g'])
```

### Phase 3: Advanced Improvements (Future)

#### Piecewise Linear Modeling for Economies of Scale
```
Current: ∂C/∂data = β₁ = constant (same cost per GB regardless of volume)
Reality: Different marginal costs by usage tiers

Example: Data cost structure
- 0-5GB: 1,000 KRW/GB (infrastructure setup costs)
- 5-20GB: 100 KRW/GB (efficiency range)  
- 20GB+: 10 KRW/GB (volume discounts)

Implementation: Slope change point detection with 1KRW/feature constraint
```

## 📊 Expected Benefits

### Immediate Improvements
1. **Accurate marginal costs**: Pure feature values without contamination
2. **Better CS ratios**: More reliable cost-spec comparisons
3. **Cleaner insights**: Each feature's true market value

### Validation Approach
1. **Consistency check**: Verify β values are economically reasonable
2. **Prediction accuracy**: Compare MAE before/after improvement
3. **Business logic**: Ensure results align with market understanding

### Backward Compatibility
- Keep existing 'frontier' method unchanged
- Add new 'multi_frontier' option
- Allow gradual migration and A/B testing

## 🎯 Mathematical Foundation

### Constraints (Same as Current)
```
1. Non-negativity: βⱼ ≥ 0 (adding features cannot reduce cost)
2. Frontier selection: Use only efficient price boundary plans
3. Minimum increment: Based on actual dataset feature differences
```

### Objective Function
```
Minimize: Σᵢ (actual_price_i - predicted_price_i)²

Where:
predicted_price_i = β₀ + β₁×data_i + β₂×voice_i + β₃×messages_i + β₄×tethering_i + β₅×5G_i

Subject to: All βⱼ ≥ 0
```

This approach combines the **frontier advantages** (overpriced plan removal) with **multi-regression advantages** (feature separation) for optimal results!

## 🚀 Implementation Plan

### Phase 1: Core Multi-Feature Regression (Immediate)
1. **Implement MultiFeatureFrontierRegression class**
   - Collect plans from all feature frontiers
   - Build complete feature matrix for regression
   - Solve constrained multi-feature regression

2. **Add new method option**
   - `calculate_cs_ratio_enhanced(method='multi_frontier')`
   - Keep existing 'frontier' method for compatibility
   - Allow A/B testing between methods

3. **Validation and testing**
   - Compare MAE between old and new methods
   - Verify economic reasonableness of β coefficients
   - Check for improved CS ratio consistency

### Phase 2: Advanced Improvements (Future)
1. **Piecewise linear modeling** for economies of scale
2. **Regularization techniques** for overfitting prevention
3. **Cross-validation** for robust coefficient estimation
4. **Interactive effects** (if data supports complexity)

### Phase 3: Production Optimization
1. **Performance optimization** for large datasets
2. **Caching mechanisms** for repeated calculations
3. **Error handling** and graceful fallbacks
4. **Documentation** and user guides

This phased approach ensures stability while enabling systematic improvements to the cost-spec analysis system.

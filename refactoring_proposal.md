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
1. **Frontier point selection**: For each feature level, select the single plan with minimum price
   - Data 10GB → cheapest 10GB plan (only one frontier point per level)
   - Data 50GB → cheapest 50GB plan (only one frontier point per level)
   - Removes overpriced plans efficiently

2. **Automatic minimum increment calculation** (modules/cost_spec.py:325-337)
   ```python
   # Calculates smallest feature value differences from actual dataset
   min_feature_increment = min(feature_differences)
   # Ensures marginal costs are based on real data increments, not arbitrary units
   ```

## ❌ Core Problem Identified: Cross-Feature Contamination in Frontier Regression

### The Real Issue
```
Example: Data feature frontier points
- 10GB frontier: Cheapest 10GB plan = [Data 10GB, Voice 200min, Messages 50, Tethering 5GB] → 30,000 KRW
- 50GB frontier: Cheapest 50GB plan = [Data 50GB, Voice ∞, Messages ∞, Tethering ∞] → 60,000 KRW

Current regression on data frontier:
- (10GB, 30,000 KRW)
- (50GB, 60,000 KRW)
- Calculated data marginal cost = (60,000 - 30,000) / (50 - 10) = 750 KRW/GB

❌ Problem: The 30,000 KRW price difference isn't just from data!
30,000 KRW difference = 40GB data + ∞voice + ∞messages + ∞tethering + other features

✅ What we want: Pure data marginal cost = ∂Price/∂Data (with other features held constant)
❌ What we get: Mixed marginal cost = ∂Price/∂(Data + Voice + Messages + Tethering + ...)
```

### Why This Happens
```
Data Frontier Points:              Voice Frontier Points:
10GB → Plan A (Voice 200min)      200min → Plan C (Data 5GB)
50GB → Plan B (Voice ∞)           ∞min → Plan D (Data 100GB)

When we calculate:
- Data marginal cost using Plan A & B → includes voice cost difference (200min vs ∞)
- Voice marginal cost using Plan C & D → includes data cost difference (5GB vs 100GB)

Result: Each feature's "marginal cost" is contaminated by other features' costs!
```

### Current Frontier Method Limitations
1. **Feature isolation**: Each feature frontier is calculated independently
2. **Cross-contamination**: Frontier points have different values for other features
3. **Regression bias**: Linear regression attributes all price differences to the target feature
4. **Inaccurate coefficients**: β values don't represent pure marginal costs

## ✅ Solution: Multi-Feature Simultaneous Regression

### Problem-Solving Approach
```
Current: Each feature → separate frontier → individual regression (contaminated)
Improved: All features → collect frontier plans → multi-feature regression (pure)

Key Insight: 
- Keep frontier selection for plan quality (remove overpriced plans)
- But perform regression on complete feature vectors to separate effects
```

### New Methodology: Frontier-Based Multi-Feature Regression

#### Step 1: Collect All Frontier Plans
```python
# Instead of separate frontiers per feature, collect all plans that appear in ANY frontier
frontier_plans = set()

# Data frontiers: 10GB→Plan A, 50GB→Plan B, 100GB→Plan C
frontier_plans.add(Plan A, Plan B, Plan C)

# Voice frontiers: 200min→Plan D, ∞→Plan E  
frontier_plans.add(Plan D, Plan E)

# Message frontiers: 50→Plan F, ∞→Plan G
frontier_plans.add(Plan F, Plan G)

# Result: {Plan A, Plan B, Plan C, Plan D, Plan E, Plan F, Plan G}
# → High-quality plan subset with diverse feature combinations
```

#### Step 2: Multi-Feature Matrix Construction
```
Plan Matrix (selected frontier plans only):
                Data   Voice   Messages  Tethering  Price
Plan A:         10GB   200min  50        5GB        30,000
Plan B:         50GB   ∞       ∞         ∞          60,000
Plan C:         100GB  500min  100       20GB       85,000
Plan D:         5GB    200min  ∞         2GB        25,000
Plan E:         15GB   ∞       50        10GB       40,000
Plan F:         20GB   300min  50        8GB        35,000
Plan G:         30GB   1000min ∞         15GB       55,000

Multi-feature regression:
Price = β₀ + β₁×Data + β₂×Voice + β₃×Messages + β₄×Tethering
```

#### Step 3: Pure Marginal Cost Extraction
```
Multi-regression result:
β₀ = 10,000 KRW (base cost)
β₁ = 400 KRW/GB (pure data cost)
β₂ = 10 KRW/min (pure voice cost)  
β₃ = 50 KRW/message (pure message cost)
β₄ = 200 KRW/GB (pure tethering cost)

Now Plan A's cost breakdown:
30,000 = 10,000 + 400×10 + 10×200 + 50×50 + 200×5
30,000 = 10,000 + 4,000 + 2,000 + 2,500 + 1,000 + remaining_features
✓ Each feature contributes its pure marginal value!
```

### Implementation: Enhanced Frontier Collection
```python
class MultiFeatureFrontierRegression:
    def collect_all_frontier_plans(self, df, features):
        """Collect plans that appear in any feature frontier"""
        frontier_plan_indices = set()
        
        for feature in features:
            # Get frontier for this feature (same as current system)
            frontier = create_robust_monotonic_frontier(df, feature, 'original_fee')
            
            # Find actual plans corresponding to frontier points
            for feature_val, min_cost in frontier.items():
                matching_plans = df[
                    (df[feature] == feature_val) & 
                    (df['original_fee'] == min_cost)
                ]
                frontier_plan_indices.update(matching_plans.index)
        
        return df.loc[list(frontier_plan_indices)]
    
    def solve_multi_feature_coefficients(self, frontier_plans, features):
        """Solve for pure marginal costs using multi-feature regression"""
        # Build feature matrix
        X = frontier_plans[features].values
        y = frontier_plans['original_fee'].values
        
        # Add intercept column
        X = np.column_stack([np.ones(len(X)), X])
        
        # Solve constrained regression (β ≥ 0)
        bounds = [(0, None)] * len(X[0])  # All coefficients non-negative
        result = minimize(
            lambda beta: np.sum((X @ beta - y) ** 2),
            x0=np.ones(len(X[0])),
            bounds=bounds
        )
        
        return result.x  # [β₀, β₁, β₂, β₃, ...]
```

## 📊 Expected Benefits

### Immediate Improvements
1. **Pure marginal costs**: Each β represents true feature value without cross-contamination
2. **Better CS ratios**: More accurate cost-spec comparisons
3. **Economic interpretability**: Coefficients align with business understanding
4. **Improved predictions**: Better fit to actual pricing patterns

### Mathematical Validation
```
Before (contaminated):
Data β₁ = 750 KRW/GB (includes voice, messages, tethering effects)

After (pure):
Data β₁ = 400 KRW/GB (pure data value only)
Voice β₂ = 10 KRW/min (pure voice value only)
Messages β₃ = 50 KRW/msg (pure message value only)

Validation check:
Mixed frontier prediction vs Multi-feature prediction accuracy
→ Multi-feature should have lower MAE and better R²
```

### Backward Compatibility
- Keep existing 'frontier' method unchanged
- Add new 'multi_frontier' option
- Allow gradual migration and A/B testing

## 🚀 Implementation Plan

### Phase 1: Core Multi-Feature Regression (Immediate)
1. **Implement frontier plan collection**
   - Extend existing frontier calculation to collect all frontier plans
   - Build unified plan matrix with complete feature vectors
   
2. **Add multi-feature regression**
   - Solve constrained optimization for pure coefficients
   - Apply same constraints as current system (β ≥ 0)

3. **Integration with existing system**
   - Add `calculate_cs_ratio_enhanced(method='multi_frontier')`
   - Maintain compatibility with current 'frontier' method

### Phase 2: Validation and Testing
1. **Mathematical validation**
   - Compare MAE between old and new methods
   - Verify economic reasonableness of coefficients
   - Check prediction accuracy on held-out data

2. **Business logic validation**
   - Ensure coefficients align with market understanding
   - Validate coefficient relationships (data > tethering, etc.)
   - Test edge cases and boundary conditions

### Phase 3: Production Deployment
1. **Performance optimization**
2. **Error handling and graceful fallbacks**
3. **Documentation and migration guides**

This approach maintains the frontier advantages (quality plan selection) while solving the cross-contamination problem through proper multi-feature regression!

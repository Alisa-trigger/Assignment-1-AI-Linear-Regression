# ============================================================
# AI Assignment No. 1 - Linear Regression for Retail Business
# IIUI — BSSE F22/F24
# REAL DATA: Sample - Superstore.csv (9,994 rows)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD REAL DATASET
# ============================================================
df = pd.read_csv('C:/Users/Princess/Downloads/Sample - Superstore.csv', encoding='latin1')

print("=" * 65)
print("  STEP 1: DATASET LOADED — Sample Superstore (Kaggle)")
print("=" * 65)
print(f"  Rows: {df.shape[0]}   |   Columns: {df.shape[1]}")
print(f"  Numerical columns used: Sales, Quantity, Discount, Profit")
print(f"\n  First 5 rows (key columns):")
print(df[['Sales','Quantity','Discount','Profit']].head().to_string())

df['Discount_Pct'] = df['Discount'] * 100

# ============================================================
# STEP 2A: DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 65)
print("  STEP 2A: DESCRIPTIVE STATISTICS")
print("=" * 65)

cols = ['Sales', 'Profit', 'Quantity', 'Discount_Pct']

for col in cols:
    s = df[col]
    print(f"\n  ── {col} ──")
    print(f"    Mean:      {s.mean():.4f}")
    print(f"    Median:    {s.median():.4f}")
    print(f"    Mode:      {s.mode()[0]:.4f}")
    print(f"    Std Dev:   {s.std():.4f}")
    print(f"    Variance:  {s.var():.4f}")
    print(f"    Min:       {s.min():.4f}")
    print(f"    Max:       {s.max():.4f}")
    print(f"    Q1 (25%):  {s.quantile(0.25):.4f}")
    print(f"    Q3 (75%):  {s.quantile(0.75):.4f}")

# ============================================================
# STEP 2B: OUTLIER DETECTION — IQR METHOD
# ============================================================
print("\n" + "=" * 65)
print("  STEP 2B: OUTLIER DETECTION (IQR METHOD)")
print("=" * 65)

def iqr_clean(df, col):
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 1.5 * IQR
    hi  = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lo) | (df[col] > hi)]
    print(f"\n  {col}:")
    print(f"    Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"    Lower fence={lo:.2f}, Upper fence={hi:.2f}")
    print(f"    Outliers detected: {len(outliers)} rows")
    return lo, hi

bounds = {}
for col in ['Sales', 'Profit', 'Quantity', 'Discount_Pct']:
    lo, hi = iqr_clean(df, col)
    bounds[col] = (lo, hi)

df_clean = df.copy()
for col, (lo, hi) in bounds.items():
    df_clean = df_clean[(df_clean[col] >= lo) & (df_clean[col] <= hi)]

print(f"\n  Rows before cleaning: {len(df)}")
print(f"  Rows after cleaning:  {len(df_clean)}")

# ============================================================
# BOXPLOTS — Outlier Visualization
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Boxplots — Outlier Detection via IQR\n(Real Superstore Data)', fontsize=13, fontweight='bold')
colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000']
for i, col in enumerate(['Sales', 'Profit', 'Quantity', 'Discount_Pct']):
    axes[i].boxplot(df[col], patch_artist=True,
                    boxprops=dict(facecolor=colors[i], alpha=0.7),
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', color='red', alpha=0.4, markersize=3))
    axes[i].set_title(col.replace('_', ' '), fontweight='bold')
    axes[i].set_ylabel('Value')
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/Princess/Downloads/boxplots_outliers.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Boxplot saved!")

# ============================================================
# STEP 2C: SCATTER PLOT ANALYSIS
# ============================================================
print("\n" + "=" * 65)
print("  STEP 2C: SCATTER PLOT ANALYSIS")
print("=" * 65)

pairs = [
    ('Sales',        'Profit',       'Profit vs Sales'),
    ('Discount_Pct', 'Profit',       'Profit vs Discount %'),
    ('Discount_Pct', 'Sales',        'Sales vs Discount %'),
    ('Quantity',     'Sales',        'Sales vs Quantity'),
    ('Quantity',     'Profit',       'Profit vs Quantity'),
]

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
fig2.suptitle('Scatter Plot Analysis — Real Superstore Data\n(All Variable Pairs with Correlation)', fontsize=14, fontweight='bold')
axes2 = axes2.flatten()

for idx, (x_col, y_col, title) in enumerate(pairs):
    ax = axes2[idx]
    x = df_clean[x_col]
    y = df_clean[y_col]
    r, p = stats.pearsonr(x, y)

    ax.scatter(x, y, alpha=0.3, s=15, color='steelblue')
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, p_line(x_line), 'r-', linewidth=2.5)

    strength = "Strong" if abs(r) > 0.6 else ("Moderate" if abs(r) > 0.3 else "Weak")
    direction = "Positive" if r > 0 else "Negative"
    ax.set_title(f'{title}\nr = {r:.3f}  |  {strength} {direction}', fontsize=10, fontweight='bold')
    ax.set_xlabel(x_col.replace('_', ' '))
    ax.set_ylabel(y_col.replace('_', ' '))
    ax.grid(True, alpha=0.3)

    print(f"  {title:35s}  r = {r:+.4f}  → {strength} {direction}")

axes2[5].set_visible(False)
plt.tight_layout()
plt.savefig('C:/Users/Princess/Downloads/scatter_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Scatter plots saved!")

# ============================================================
# STEP 3: LINEAR REGRESSION — LINE OF BEST FIT
# ============================================================
print("\n" + "=" * 65)
print("  STEP 3: REGRESSION MODELS")
print("=" * 65)

regression_pairs = [
    ('Sales',        'Profit',    'Profit vs Sales'),
    ('Discount_Pct', 'Profit',    'Profit vs Discount %'),
    ('Discount_Pct', 'Sales',     'Sales vs Discount %'),
    ('Quantity',     'Sales',     'Sales vs Quantity'),
    ('Quantity',     'Profit',    'Profit vs Quantity'),
]

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 11))
fig3.suptitle('Linear Regression — Lines of Best Fit\n(Real Superstore Data)', fontsize=14, fontweight='bold')
axes3 = axes3.flatten()

models = {}
for idx, (x_col, y_col, label) in enumerate(regression_pairs):
    X = df_clean[[x_col]].values
    Y = df_clean[y_col].values

    model = LinearRegression()
    model.fit(X, Y)
    slope     = model.coef_[0]
    intercept = model.intercept_
    Y_pred    = model.predict(X)
    r2        = r2_score(Y, Y_pred)
    rmse      = np.sqrt(mean_squared_error(Y, Y_pred))

    models[label] = dict(model=model, slope=slope, intercept=intercept,
                         r2=r2, rmse=rmse, x_col=x_col, y_col=y_col)

    print(f"\n  ── {label} ──")
    print(f"    Equation:  {y_col} = {slope:.4f} × {x_col} + {intercept:.4f}")
    print(f"    R²:        {r2:.4f}  ({r2*100:.1f}% variance explained)")
    print(f"    RMSE:      {rmse:.4f}")
    sign = "increases" if slope > 0 else "decreases"
    print(f"    → For every 1-unit increase in {x_col.replace('_',' ')}, {y_col} {sign} by {abs(slope):.4f}")

    ax = axes3[idx]
    ax.scatter(X, Y, alpha=0.3, s=15, color='steelblue', label='Actual data')
    ax.plot(X, Y_pred, 'r-', linewidth=2.5, label='Best fit line')
    ax.set_xlabel(x_col.replace('_', ' '))
    ax.set_ylabel(y_col.replace('_', ' '))
    ax.set_title(f'{label}\nY = {slope:.3f}X + {intercept:.1f}  |  R²={r2:.3f}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes3[5].set_visible(False)
plt.tight_layout()
plt.savefig('C:/Users/Princess/Downloads/regression_lines.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Regression plots saved!")

# ============================================================
# CORRELATION HEATMAP
# ============================================================
fig4, ax4 = plt.subplots(figsize=(8, 6))
corr = df_clean[['Sales','Profit','Quantity','Discount_Pct']].corr()
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            ax=ax4, linewidths=1, annot_kws={'size': 13, 'weight': 'bold'})
ax4.set_title('Correlation Heatmap — Real Superstore Data', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('C:/Users/Princess/Downloads/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Correlation heatmap saved!")

# ============================================================
# STEP 4: PREDICTION SCENARIOS
# ============================================================
print("\n" + "=" * 65)
print("  STEP 4: PREDICTION SCENARIOS")
print("=" * 65)

def predict(label, x_val):
    return models[label]['model'].predict([[x_val]])[0]

m = models

print("\n  Scenario 1: What profit is expected from $500 in Sales?")
p1 = predict('Profit vs Sales', 500)
s1 = m['Profit vs Sales']['slope']
i1 = m['Profit vs Sales']['intercept']
print(f"    Profit = {s1:.4f} × 500 + {i1:.4f} = ${p1:.2f}")
print(f"    → A $500 sale is expected to generate ${p1:.2f} in profit.")

print("\n  Scenario 2: How does a 20% discount affect profit?")
p2a = predict('Profit vs Discount %', 20)
p2b = predict('Profit vs Discount %', 0)
s2  = m['Profit vs Discount %']['slope']
i2  = m['Profit vs Discount %']['intercept']
print(f"    At 0%  discount: Profit = {s2:.4f}×0  + {i2:.4f} = ${p2b:.2f}")
print(f"    At 20% discount: Profit = {s2:.4f}×20 + {i2:.4f} = ${p2a:.2f}")
print(f"    → A 20% discount reduces profit by ${p2b - p2a:.2f} per transaction.")

print("\n  Scenario 3: Predict sales for quantities 10, 50, and 100 units")
for qty in [10, 50, 100]:
    sp = predict('Sales vs Quantity', qty)
    print(f"    Quantity = {qty:3d} → Predicted Sales = ${sp:.2f}")

print("\n  Scenario 4: What discount level kills all profit? (Break-even)")
slope_d  = m['Profit vs Discount %']['slope']
inter_d  = m['Profit vs Discount %']['intercept']
breakeven = -inter_d / slope_d
print(f"    Profit = 0 when Discount = {breakeven:.1f}%")
print(f"    → Any discount above {breakeven:.1f}% results in a LOSS.")

# ============================================================
# STEP 5: BUSINESS STRATEGY RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 65)
print("  STEP 5: BUSINESS STRATEGY RECOMMENDATIONS")
print("=" * 65)

strategies = [
    ("1. Protect Profit Margins — Control Discounting",
     f"Discount has a NEGATIVE effect on profit (slope={m['Profit vs Discount %']['slope']:.2f}).\n"
     f"    Break-even discount = {breakeven:.1f}%. Never exceed this.\n"
     f"    → Replace blanket discounts with loyalty-member-only deals.\n"
     f"    → Use bundle offers instead of % discounts."),

    ("2. Sales-Profit Relationship — Grow Revenue Smartly",
     f"Every $1 increase in sales adds ${m['Profit vs Sales']['slope']:.4f} to profit.\n"
     f"    → Focus on high-value categories: Furniture, Technology.\n"
     f"    → Upsell complementary products to raise average order value."),

    ("3. Quantity Strategy — Volume = Revenue",
     f"Higher quantity sold means higher sales.\n"
     f"    → Offer bulk-buy deals to push quantity up.\n"
     f"    → Never let fast-moving items go out of stock."),

    ("4. Discount is NOT driving Sales",
     f"Discount shows only weak effect on sales (r = {stats.pearsonr(df_clean['Discount_Pct'], df_clean['Sales'])[0]:.3f}).\n"
     f"    → Discounts hurt profit WITHOUT reliably boosting sales.\n"
     f"    → Redirect discount budget into better product placement."),
]

for title, detail in strategies:
    print(f"\n  ★ {title}")
    print(f"    {detail}")

print("\n\n" + "=" * 65)
print("  ALL DONE! Check your Downloads folder for the 4 chart images.")
print("=" * 65)
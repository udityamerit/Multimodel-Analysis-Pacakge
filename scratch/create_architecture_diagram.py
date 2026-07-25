import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram(output_path="architecture_diagram.png"):
    # Set up figure
    fig, ax = plt.subplots(figsize=(20, 10.5), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis('off')

    # Title Banner
    ax.text(50, 48.6, "MultiModel Analysis — End-to-End System Architecture", 
            fontsize=24, fontweight='bold', color='#FFFFFF', ha='center', va='center',
            fontfamily='sans-serif')
    ax.text(50, 46.3, "Automated Model Benchmarking, Diagnostic Evaluation & Visualization Pipeline for Scikit-Learn", 
            fontsize=13, color='#94A3B8', ha='center', va='center', fontfamily='sans-serif')

    # Decorative Top Gradient Line
    ax.plot([10, 90], [44.8, 44.8], color='#38BDF8', lw=2.5, alpha=0.8)

    # Box drawer helper
    def draw_card(x, y, w, h, title, subtitle="", items=[], bg_color='#0F172A', border_color='#38BDF8', header_color='#38BDF8', step_y=1.8):
        # Card Background Shadow
        shadow = patches.FancyBboxPatch((x+0.4, y-0.4), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                        facecolor='#020617', edgecolor='none', alpha=0.5, zorder=2)
        ax.add_patch(shadow)
        
        # Main Card Body
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                      facecolor=bg_color, edgecolor=border_color, linewidth=1.8, zorder=3)
        ax.add_patch(card)
        
        # Header Box
        header_box = patches.FancyBboxPatch((x, y + h - 3.2), w, 3.2, boxstyle="round,pad=0.5,rounding_size=0.8",
                                            facecolor=header_color, edgecolor='none', zorder=4)
        ax.add_patch(header_box)
        
        # Header Text
        ax.text(x + w/2, y + h - 1.6, title, fontsize=12, fontweight='bold', color='#0F172A',
                ha='center', va='center', zorder=5, fontfamily='sans-serif')
        
        # Subtitle / Body text
        curr_y = y + h - 5.0
        if subtitle:
            ax.text(x + w/2, curr_y, subtitle, fontsize=10, fontweight='bold', color='#E2E8F0',
                    ha='center', va='center', zorder=5, fontfamily='sans-serif')
            curr_y -= 2.2
            
        for item in items:
            if not item:
                curr_y -= 1.0
                continue
            is_section = item.endswith(":")
            font_w = 'bold' if is_section else 'normal'
            font_c = '#38BDF8' if is_section else '#CBD5E1'
            bullet = "" if is_section else "• "
            ax.text(x + 1.0, curr_y, f"{bullet}{item}", fontsize=9.0, fontweight=font_w, color=font_c,
                    ha='left', va='center', zorder=5, fontfamily='sans-serif')
            curr_y -= step_y

    # Connector Arrow Helper
    def draw_arrow(x1, y1, x2, y2, color='#38BDF8', label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=2.5, headwidth=9, headlength=10, shrink=0.05),
                    zorder=6)
        if label:
            mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mid_x, mid_y + 1.2, label, fontsize=8.5, fontweight='bold', color=color,
                    ha='center', va='center', zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#0F172A', edgecolor=color, lw=0.8))

    # Stage 1: Input Data
    draw_card(3, 24, 15, 17, 
              title="1. INPUT DATASET", 
              subtitle="Raw Features & Targets",
              items=["Features (X): DataFrame/Array", "Target (y): 1D/2D Array", "Target Flattening (.ravel())", "Column Metadata Preserved"],
              bg_color='#0F172A', border_color='#0284C7', header_color='#38BDF8', step_y=2.0)

    # Arrow 1 -> 2
    draw_arrow(18.5, 32.5, 23.5, 32.5, label="Raw Arrays")

    # Stage 2: Preprocessing & Scaling
    draw_card(24, 24, 16, 17, 
              title="2. PREPROCESSING", 
              subtitle="Encoding & Scaling",
              items=["LabelEncoder for Targets", "StandardScaler (Optional)", "Index & Column Retention", "Non-Mutating Warning Scope"],
              bg_color='#0F172A', border_color='#818CF8', header_color='#A5B4FC', step_y=2.0)

    # Arrow 2 -> 3
    draw_arrow(40.5, 32.5, 45.5, 32.5, label="Scaled Data")

    # Stage 3: Train-Test Splitter
    draw_card(46, 24, 15, 17, 
              title="3. DATA SPLITTER", 
              subtitle="Train / Test Partitioning",
              items=["train_test_split()", "Stratified Split Guard", "(min_class_count >= 2)", "Configurable test_size", "Random State Reproducibility"],
              bg_color='#0F172A', border_color='#F59E0B', header_color='#FDE047', step_y=2.0)

    # Arrow 3 -> 4 (Splits into Classification & Regression)
    draw_arrow(61.5, 36, 66.5, 37.5, label="Classification", color='#34D399')
    draw_arrow(61.5, 29, 66.5, 25.5, label="Regression", color='#F472B6')

    # Stage 4A: MultiModelClassifier Engine
    draw_card(67, 28.5, 15, 14, 
              title="4A. CLASSIFIER SUITE", 
              subtitle="8 Baseline + Custom Models",
              items=["Logistic, SVM, DecisionTree", "KNN, NaiveBayes, RandomForest", "GradientBoosting, AdaBoost", "Custom User Estimators", "Parallel Execution (n_jobs)"],
              bg_color='#0F172A', border_color='#10B981', header_color='#6EE7B7', step_y=1.8)

    # Stage 4B: MultiModelRegressor Engine
    draw_card(67, 12, 15, 14.5, 
              title="4B. REGRESSOR SUITE", 
              subtitle="7 Baseline + Custom Models",
              items=["Linear, Lasso, Ridge", "SVR (RBF Kernel)", "DecisionTree, RandomForest", "GradientBoosting Regressor", "Custom Regressor Support"],
              bg_color='#0F172A', border_color='#EC4899', header_color='#F472B6', step_y=1.8)

    # Arrow 4A & 4B -> 5
    draw_arrow(82.5, 35.5, 85.5, 27, label="Class Metrics", color='#34D399')
    draw_arrow(82.5, 19, 85.5, 22, label="Reg Metrics", color='#F472B6')

    # Stage 5: Evaluation & Output Engine
    draw_card(86, 11, 12.5, 32.5, 
              title="5. OUTPUT ENGINE", 
              subtitle="Reporting & Charts",
              items=[
                  "Metrics Engine:",
                  "Acc, Prec, Rec, F1",
                  "Macro ROC-AUC",
                  "MAE, MSE, RMSE, R²",
                  "",
                  "Visual Diagnostics:",
                  "Confusion Matrices",
                  "ROC Curves (Macro/Binary)",
                  "True vs Pred Scatter",
                  "Metric Comparison Bar",
                  "",
                  "File Exporter:",
                  "CSV, Excel, HTML, JSON",
                  "PNG Figures (save_path)"
              ],
              bg_color='#0F172A', border_color='#C084FC', header_color='#E9D5FF', step_y=1.5)

    # Bottom Pipeline Workflow Footer Box
    footer_box = patches.FancyBboxPatch((3, 2.5), 95, 6.5, boxstyle="round,pad=0.5,rounding_size=1.0",
                                        facecolor='#1E293B', edgecolor='#334155', linewidth=1.2, zorder=3)
    ax.add_patch(footer_box)
    
    ax.text(50, 6.6, "Key Features in v0.1.0+", fontsize=11, fontweight='bold', color='#38BDF8',
            ha='center', va='center', zorder=5, fontfamily='sans-serif')
    
    features_text = "✓ Macro-Average & Per-Class ROC Curves   |   ✓ Safe NaN Fallback for Imbalanced Classes   |   ✓ Non-Mutating Matplotlib & Warning Scoping   |   ✓ Single-Line Auto-Report (get_summary)"
    ax.text(50, 4.3, features_text, fontsize=9.5, color='#E2E8F0', ha='center', va='center', zorder=5, fontfamily='sans-serif')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0B0F19')
    plt.close()
    print(f"Architecture diagram regenerated successfully: {output_path}")

if __name__ == '__main__':
    draw_architecture_diagram()

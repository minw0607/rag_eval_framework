"""
Results visualization for the RAG evaluation framework.

Generates three publication-quality figures:
  Figure 1 — Main evaluation dashboard (6 panels)
  Figure 2 — Retrieval quality analysis (2 panels)
  Figure 3 — Cost & cascade analysis (3 panels)

Functions
---------
generate_dashboard(results_file, output_dir) -> None
"""

import glob
import json
import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats


# =============================================================================
# Public API
# =============================================================================

def generate_dashboard(results_file: str, output_dir: str) -> None:
    """
    Generate all three visualization figures and save them to output_dir.

    Figures produced
    ----------------
    evaluation_dashboard.png    — 6-panel main dashboard
    retrieval_analysis.png      — 2-panel retrieval quality
    cost_cascade_analysis.png   — 3-panel cost & cascade statistics

    Parameters
    ----------
    results_file : str
        Path to the JSON results file produced by run_evaluation().
        If empty string or None, the most-recently-modified results file
        matching ``output_dir/multi_prompt_eval_results_*.json`` is used.
    output_dir : str
        Directory where figures are saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- Load results ---------------------------------------------------------
    if not results_file:
        candidates = sorted(
            glob.glob(os.path.join(output_dir, 'multi_prompt_eval_results_*.json')),
            key=os.path.getmtime
        )
        if not candidates:
            raise FileNotFoundError(f"No results files found in '{output_dir}/'")
        results_file = candidates[-1]

    print(f"📂 Loading: {results_file}")

    with open(results_file, 'r') as f:
        viz_eval_data = json.load(f)

    viz_metadata   = viz_eval_data['metadata']
    viz_aggregated = viz_eval_data['aggregated_results']
    viz_detailed   = viz_eval_data['detailed_results']
    viz_variants   = list(viz_aggregated.keys())
    viz_n          = viz_metadata['num_questions']

    print(f"✅ Loaded {viz_n} questions | Variants: {', '.join(viz_variants)}")
    print(f"   Faithfulness method: {viz_metadata.get('faithfulness_method', 'Unknown')}")

    # -- Shared config --------------------------------------------------------
    VARIANT_COLORS = {
        'baseline': '#3498db',
        'concise':  '#2ecc71',
        'detailed': '#e67e22',
        'citation': '#9b59b6',
    }

    METRIC_THRESHOLDS = {
        'f1':           0.70,
        'groundedness': 0.50,
        'completeness': 0.40,
        'faithfulness': 0.70,
        'ans_rel':      0.50,
        'ctx_rel':      0.40,
        'quality':      0.70,
    }

    def get_color(variant):
        return VARIANT_COLORS.get(variant, '#95a5a6')

    def threshold_color(val, thresh):
        if val is None:
            return '#cccccc'
        if val >= thresh * 1.10:
            return '#27ae60'
        elif val >= thresh:
            return '#f39c12'
        return '#e74c3c'

    viz_hit_rates = [r['hit_rate'] for r in viz_detailed]
    viz_avg_hit   = np.mean(viz_hit_rates)

    # =========================================================================
    # FIGURE 1: MAIN EVALUATION DASHBOARD (6 panels)
    # =========================================================================
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f'RAG Multi-Prompt Evaluation Dashboard\n'
        f'HotpotQA | {viz_metadata.get("embedding_model", "N/A")} | '
        f'{viz_n:,} Questions | '
        f'Faithfulness: {viz_metadata.get("faithfulness_method", "N/A")}',
        fontsize=13, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.42)

    # Panel 1: F1≥0.5 Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    viz_f1_vals = [viz_aggregated[v]['accuracy_f1'] for v in viz_variants]
    viz_bars1 = ax1.bar(
        [v.capitalize() for v in viz_variants],
        viz_f1_vals,
        color=[get_color(v) for v in viz_variants],
        alpha=0.88, edgecolor='white', linewidth=1.2
    )
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel('F1 ≥ 0.5 Accuracy', fontsize=9)
    ax1.set_title('Accuracy by Prompt Variant', fontweight='bold', fontsize=10)
    ax1.axhline(y=0.70, color='#7f8c8d', linestyle='--', alpha=0.6, linewidth=1.2)
    ax1.text(len(viz_variants) - 0.5, 0.72, '70% target', fontsize=7, color='#7f8c8d', ha='right')
    for bar, val in zip(viz_bars1, viz_f1_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.026,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel 2: Quality Metrics Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    viz_heatmap_metrics = [
        ('avg_groundedness',      'Ground',  0.50),
        ('avg_completeness',      'Complet', 0.40),
        ('avg_faithfulness',      'Faith',   0.70),
        ('avg_answer_relevancy',  'AnsRel',  0.50),
        ('avg_context_relevancy', 'CtxRel',  0.40),
        ('avg_quality_score',     'Quality', 0.70),
    ]
    viz_heatmap_data = np.array([
        [viz_aggregated[v].get(metric) or 0 for metric, _, _ in viz_heatmap_metrics]
        for v in viz_variants
    ])
    viz_im = ax2.imshow(viz_heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    for i in range(len(viz_variants)):
        for j, (_, _, thresh) in enumerate(viz_heatmap_metrics):
            val      = viz_heatmap_data[i, j]
            txt_color = 'white' if val < 0.35 or val > 0.82 else 'black'
            badge    = '✓' if val >= thresh else '⚠'
            ax2.text(j, i, f'{val:.2f}\n{badge}',
                     ha='center', va='center', fontsize=7.5,
                     color=txt_color, fontweight='bold')
    ax2.set_xticks(range(len(viz_heatmap_metrics)))
    ax2.set_xticklabels([lbl for _, lbl, _ in viz_heatmap_metrics],
                         fontsize=8, rotation=30, ha='right')
    ax2.set_yticks(range(len(viz_variants)))
    ax2.set_yticklabels([v.capitalize() for v in viz_variants], fontsize=9)
    ax2.set_title('Quality Metrics Heatmap\n(✓=above threshold, ⚠=below)',
                  fontweight='bold', fontsize=9)
    viz_cbar = plt.colorbar(viz_im, ax=ax2, shrink=0.78, pad=0.02)
    viz_cbar.set_label('Score', fontsize=7)
    viz_cbar.ax.tick_params(labelsize=7)

    # Panel 3: Contains vs F1 Scatter
    ax3 = fig.add_subplot(gs[0, 2])
    label_offsets = {'baseline': (8, -12), 'citation': (8, 6), 'concise': (-62, 6), 'detailed': (8, 6)}
    for variant in viz_variants:
        ax3.scatter(
            viz_aggregated[variant]['accuracy_contains'],
            viz_aggregated[variant]['accuracy_f1'],
            color=get_color(variant), s=220, zorder=5,
            label=variant.capitalize(), edgecolors='white', linewidths=1.5
        )
        ax3.annotate(
            variant.capitalize(),
            (viz_aggregated[variant]['accuracy_contains'], viz_aggregated[variant]['accuracy_f1']),
            textcoords='offset points',
            xytext=label_offsets.get(variant, (6, 4)),
            fontsize=8.5, fontweight='bold', color=get_color(variant)
        )
    ax3.set_xlabel('Contains Accuracy', fontsize=9)
    ax3.set_ylabel('F1 ≥ 0.5 Accuracy', fontsize=9)
    ax3.set_title('Contains vs F1 Accuracy\n(above diagonal = concise answers)',
                  fontweight='bold', fontsize=9)
    ax3.set_xlim(0, 1.1)
    ax3.set_ylim(0, 1.1)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1.2)
    ax3.grid(True, alpha=0.22)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Panel 4: Faithfulness Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    for variant in viz_variants:
        viz_faith_dist = [
            r['variant_results'][variant].get('faithfulness')
            for r in viz_detailed
            if r['variant_results'][variant].get('faithfulness') is not None
        ]
        if viz_faith_dist:
            ax4.hist(viz_faith_dist, bins=15, histtype='step', linewidth=2,
                     color=get_color(variant),
                     label=f'{variant.capitalize()} (μ={np.mean(viz_faith_dist):.2f})')
    ax4.set_xlabel('Faithfulness Score (RAGAS LLM-as-judge)', fontsize=9)
    ax4.set_ylabel('Count', fontsize=9)
    ax4.set_title('Faithfulness Score Distribution', fontweight='bold', fontsize=10)
    ax4.legend(fontsize=7.5)
    ax4.axvline(x=0.70, color='#7f8c8d', linestyle='--', alpha=0.7, linewidth=1.2)
    ylim = ax4.get_ylim()
    ax4.text(0.71, ylim[1] * 0.85 if ylim[1] > 0 else 10,
             'Threshold\n(0.70)', fontsize=7, color='#7f8c8d')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Panel 5: Refusal & Citation Rates
    ax5  = fig.add_subplot(gs[1, 1])
    viz_x5   = np.arange(len(viz_variants))
    viz_w5   = 0.35
    viz_ref  = [viz_aggregated[v]['refusal_rate']  for v in viz_variants]
    viz_cite = [viz_aggregated[v]['citation_rate'] for v in viz_variants]
    ax5.bar(viz_x5 - viz_w5 / 2, viz_ref,  viz_w5, label='Refusal Rate',
            color='#e74c3c', alpha=0.85, edgecolor='white')
    ax5.bar(viz_x5 + viz_w5 / 2, viz_cite, viz_w5, label='Citation Rate',
            color='#27ae60', alpha=0.85, edgecolor='white')
    ax5.set_xticks(viz_x5)
    ax5.set_xticklabels([v.capitalize() for v in viz_variants], fontsize=9)
    ax5.set_ylim(0, 1.20)
    ax5.set_ylabel('Rate', fontsize=9)
    ax5.set_title('Refusal & Citation Rates', fontweight='bold', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.axhline(y=0.10, color='#e74c3c', linestyle=':', alpha=0.4, linewidth=1)
    ax5.text(len(viz_variants) - 0.5, 0.115, '10% refusal cap',
             fontsize=6.5, color='#e74c3c', ha='right')
    ax5.axhline(y=0.90, color='#27ae60', linestyle=':', alpha=0.4, linewidth=1)
    ax5.text(0, 0.915, '90% cite target', fontsize=6.5, color='#27ae60')
    for i, (ref, cit) in enumerate(zip(viz_ref, viz_cite)):
        ax5.text(i - viz_w5 / 2, ref + 0.025, f'{ref:.0%}',
                 ha='center', fontsize=8, fontweight='bold')
        ax5.text(i + viz_w5 / 2, cit + 0.025, f'{cit:.0%}',
                 ha='center', fontsize=8, fontweight='bold')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Panel 6: Summary Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    viz_tbl_headers = ['Variant', 'F1≥0.5', 'Contains', 'Ground',
                       'Faith', 'Quality', 'Cite', 'Refusal', 'Len']
    viz_tbl_rows = []
    for variant in viz_variants:
        m = viz_aggregated[variant]
        viz_tbl_rows.append([
            variant.capitalize(),
            f"{m['accuracy_f1']:.1%}",
            f"{m['accuracy_contains']:.1%}",
            f"{m.get('avg_groundedness') or 0:.2f}",
            f"{m.get('avg_faithfulness') or 0:.2f}",
            f"{m.get('avg_quality_score') or 0:.2f}",
            f"{m['citation_rate']:.0%}",
            f"{m['refusal_rate']:.1%}",
            f"{m['avg_answer_length']:.0f}w",
        ])
    viz_tbl_rows.append(['─' * 5] * 9)
    viz_tbl_rows.append(['HitRate', f'{viz_avg_hit:.1%}', '─', '─', '─', '─', '─', '─', '─'])
    if viz_metadata.get('total_cost'):
        viz_tbl_rows.append([
            'Cost', f"${viz_metadata['total_cost']:.2f}",
            f"${viz_metadata['total_cost'] / viz_n:.4f}/q",
            '─', '─', '─', '─', '─', '─',
        ])
    viz_tbl = ax6.table(
        cellText=viz_tbl_rows, colLabels=viz_tbl_headers,
        loc='center', cellLoc='center'
    )
    viz_tbl.auto_set_font_size(False)
    viz_tbl.set_fontsize(7.5)
    viz_tbl.scale(1.0, 1.52)
    for (row, col), cell in viz_tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif 1 <= row <= len(viz_variants) and col == 0:
            cell.set_facecolor(get_color(viz_variants[row - 1]) + '40')
        elif row % 2 == 0:
            cell.set_facecolor('#f4f6f8')
    ax6.set_title('Summary Table', fontweight='bold', pad=12, fontsize=10)

    viz_path1 = os.path.join(output_dir, 'evaluation_dashboard.png')
    plt.savefig(viz_path1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Figure 1 saved: {viz_path1}")

    # =========================================================================
    # FIGURE 2: RETRIEVAL QUALITY ANALYSIS (2 panels)
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
    fig2.suptitle('Retrieval Quality Analysis', fontsize=13, fontweight='bold')

    ax_hit = axes2[0]
    ax_hit.hist(viz_hit_rates, bins=20, color='#3498db', alpha=0.85,
                edgecolor='white', linewidth=0.8)
    ax_hit.axvline(x=viz_avg_hit, color='#e74c3c', linestyle='--',
                   linewidth=2, label=f'Mean: {viz_avg_hit:.1%}')
    ax_hit.axvline(x=0.70, color='#7f8c8d', linestyle=':', linewidth=1.5,
                   alpha=0.7, label='70% target')
    ax_hit.set_xlabel('Hit Rate', fontsize=9)
    ax_hit.set_ylabel('Count', fontsize=9)
    ax_hit.set_title('Retrieval Hit Rate Distribution', fontweight='bold')
    ax_hit.legend(fontsize=9)
    ax_hit.spines['top'].set_visible(False)
    ax_hit.spines['right'].set_visible(False)
    viz_perfect = sum(1 for h in viz_hit_rates if h == 1.0)
    viz_zero    = sum(1 for h in viz_hit_rates if h == 0.0)
    ax_hit.text(0.05, 0.92,
                f'Perfect (100%): {viz_perfect} ({viz_perfect / viz_n:.0%})\n'
                f'Zero (0%):  {viz_zero} ({viz_zero / viz_n:.0%})',
                transform=ax_hit.transAxes, fontsize=8, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax_corr = axes2[1]
    for variant in viz_variants:
        viz_hr_list = [r['hit_rate'] for r in viz_detailed]
        viz_f1_list = [r['variant_results'][variant].get('f1_score', 0) for r in viz_detailed]
        ax_corr.scatter(viz_hr_list, viz_f1_list, color=get_color(variant), alpha=0.10, s=12)
        viz_slope, viz_intercept, viz_r, _, _ = stats.linregress(viz_hr_list, viz_f1_list)
        viz_x_line = np.linspace(min(viz_hr_list), max(viz_hr_list), 100)
        ax_corr.plot(viz_x_line, viz_slope * viz_x_line + viz_intercept,
                     color=get_color(variant), linewidth=2.5,
                     label=f'{variant.capitalize()} (r={viz_r:.2f})')
    ax_corr.set_xlabel('Retrieval Hit Rate', fontsize=9)
    ax_corr.set_ylabel('F1 Score', fontsize=9)
    ax_corr.set_title('Hit Rate vs F1 Score\n(trend lines per variant)', fontweight='bold')
    ax_corr.legend(fontsize=8.5)
    ax_corr.grid(True, alpha=0.22)
    ax_corr.spines['top'].set_visible(False)
    ax_corr.spines['right'].set_visible(False)

    plt.tight_layout()
    viz_path2 = os.path.join(output_dir, 'retrieval_analysis.png')
    plt.savefig(viz_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Figure 2 saved: {viz_path2}")

    # =========================================================================
    # FIGURE 3: COST & CASCADE ANALYSIS (3 panels)
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle(
        f'Cost & Cascade Evaluation Analysis\n'
        f'{viz_n:,} Questions | Completeness: Cascade '
        f'(Tier 1 Embedding + Tier 2 LLM)',
        fontsize=12, fontweight='bold'
    )

    viz_cost_meta  = viz_metadata.get('cost_summary', {})
    viz_gen_costs  = viz_cost_meta.get('generation_costs',  {}) if viz_cost_meta else {}
    viz_eval_costs = viz_cost_meta.get('evaluation_costs',  {}) if viz_cost_meta else {}

    viz_cost_display = {
        'llm_answer_generation':  'Answer Gen (LLM)',
        'embedding_retrieval':    'Retrieval Embed',
        'embedding_completeness': 'Completeness Embed',
        'embedding_groundedness': 'Groundedness Embed',
        'llm_judge_faithfulness': 'Faith Judge (LLM)',
        'llm_judge_completeness': 'Complete Judge (LLM)',
    }
    viz_gen_colors = {'llm_answer_generation': '#2980b9'}
    viz_eval_colors = {
        'embedding_retrieval':    '#27ae60',
        'embedding_completeness': '#2ecc71',
        'embedding_groundedness': '#82e0aa',
        'llm_judge_faithfulness': '#e67e22',
        'llm_judge_completeness': '#e74c3c',
    }

    ax_cost = axes3[0]
    if viz_gen_costs or viz_eval_costs:
        viz_labels, viz_vals, viz_cols = [], [], []
        for lbl, val in viz_gen_costs.items():
            if val > 0:
                viz_labels.append(viz_cost_display.get(lbl, lbl))
                viz_vals.append(val)
                viz_cols.append(viz_gen_colors.get(lbl, '#3498db'))
        for lbl, val in viz_eval_costs.items():
            if val > 0:
                viz_labels.append(viz_cost_display.get(lbl, lbl))
                viz_vals.append(val)
                viz_cols.append(viz_eval_colors.get(lbl, '#95a5a6'))
        viz_cost_bars = ax_cost.barh(viz_labels, viz_vals, color=viz_cols,
                                      alpha=0.85, edgecolor='white')
        for bar, val in zip(viz_cost_bars, viz_vals):
            ax_cost.text(bar.get_width() + max(viz_vals) * 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f'${val:.3f}', va='center', fontsize=8)
        gen_patch  = mpatches.Patch(color='#2980b9', alpha=0.85, label='Generation (production)')
        eval_patch = mpatches.Patch(color='#e67e22', alpha=0.85, label='Evaluation (framework)')
        ax_cost.legend(handles=[gen_patch, eval_patch], fontsize=7.5, loc='lower right')
        viz_tgen  = sum(viz_gen_costs.values())
        viz_teval = sum(viz_eval_costs.values())
        ax_cost.set_title(
            f'Cost Breakdown — Gen: ${viz_tgen:.2f} | Eval: ${viz_teval:.2f}\n'
            f'(${(viz_tgen + viz_teval) / viz_n:.4f}/question)',
            fontweight='bold', fontsize=9
        )
        ax_cost.set_xlabel('Cost (USD)', fontsize=9)
        ax_cost.spines['top'].set_visible(False)
        ax_cost.spines['right'].set_visible(False)
    else:
        ax_cost.text(0.5, 0.5,
                     'Cost summary not available\n(run with ENABLE_COST_TRACKING=True)',
                     ha='center', va='center', transform=ax_cost.transAxes,
                     fontsize=10, color='gray')
        ax_cost.set_title('Cost Breakdown', fontweight='bold')
        ax_cost.axis('off')

    ax_cascade = axes3[1]
    viz_cx    = np.arange(len(viz_variants))
    viz_cw    = 0.35
    viz_trigs = [viz_aggregated[v].get('cascade_trigger_rate') or 0 for v in viz_variants]
    viz_agrs  = [viz_aggregated[v].get('cascade_agreement_rate') or 0 for v in viz_variants]
    ax_cascade.bar(viz_cx - viz_cw / 2, viz_trigs, viz_cw,
                   label='Trigger Rate (T2 fired)',
                   color='#e74c3c', alpha=0.85, edgecolor='white')
    ax_cascade.bar(viz_cx + viz_cw / 2, viz_agrs, viz_cw,
                   label='Agreement Rate (T1≈T2)',
                   color='#27ae60', alpha=0.85, edgecolor='white')
    ax_cascade.set_xticks(viz_cx)
    ax_cascade.set_xticklabels([v.capitalize() for v in viz_variants], fontsize=9)
    ax_cascade.set_ylim(0, 1.20)
    ax_cascade.set_ylabel('Rate', fontsize=9)
    ax_cascade.set_title(
        'Completeness Cascade Statistics\n(Tier 2 LLM trigger & T1/T2 agreement)',
        fontweight='bold', fontsize=9
    )
    ax_cascade.legend(fontsize=8)
    ax_cascade.axhline(y=0.20, color='#7f8c8d', linestyle='--', alpha=0.5, linewidth=1)
    ax_cascade.text(len(viz_variants) - 0.5, 0.22, '20% target',
                    fontsize=7, color='#7f8c8d', ha='right')
    for i, (trig, agr) in enumerate(zip(viz_trigs, viz_agrs)):
        ax_cascade.text(i - viz_cw / 2, trig + 0.025, f'{trig:.0%}',
                        ha='center', fontsize=8, fontweight='bold')
        ax_cascade.text(i + viz_cw / 2, agr + 0.025, f'{agr:.0%}',
                        ha='center', fontsize=8, fontweight='bold')
    ax_cascade.spines['top'].set_visible(False)
    ax_cascade.spines['right'].set_visible(False)

    ax_comp  = axes3[2]
    viz_t1   = [viz_aggregated[v].get('avg_context_coverage') or 0 for v in viz_variants]
    viz_cas  = [viz_aggregated[v].get('avg_completeness')     or 0 for v in viz_variants]
    viz_qx   = np.arange(len(viz_variants))
    viz_qw   = 0.35
    ax_comp.bar(viz_qx - viz_qw / 2, viz_t1, viz_qw,
                label='Context Coverage (Tier 1 embedding)',
                color='#95a5a6', alpha=0.85, edgecolor='white')
    ax_comp.bar(viz_qx + viz_qw / 2, viz_cas, viz_qw,
                label='Completeness (Cascade final)',
                color=[get_color(v) for v in viz_variants],
                alpha=0.85, edgecolor='white')
    ax_comp.set_xticks(viz_qx)
    ax_comp.set_xticklabels([v.capitalize() for v in viz_variants], fontsize=9)
    ax_comp.set_ylim(0, 1.15)
    ax_comp.set_ylabel('Score', fontsize=9)
    ax_comp.set_title(
        'Context Coverage vs Completeness\n(Tier 1 embedding → Cascade final)',
        fontweight='bold', fontsize=9
    )
    ax_comp.legend(fontsize=7.5)
    ax_comp.axhline(y=0.40, color='#7f8c8d', linestyle='--', alpha=0.4, linewidth=1)
    ax_comp.text(len(viz_variants) - 0.5, 0.42, 'threshold (0.40)',
                 fontsize=6.5, color='#7f8c8d', ha='right')
    for i, (t1v, casv) in enumerate(zip(viz_t1, viz_cas)):
        ax_comp.text(i - viz_qw / 2, t1v + 0.025, f'{t1v:.2f}',
                     ha='center', fontsize=8, color='#555')
        ax_comp.text(i + viz_qw / 2, casv + 0.025, f'{casv:.2f}',
                     ha='center', fontsize=8, fontweight='bold')
    ax_comp.spines['top'].set_visible(False)
    ax_comp.spines['right'].set_visible(False)

    plt.tight_layout()
    viz_path3 = os.path.join(output_dir, 'cost_cascade_analysis.png')
    plt.savefig(viz_path3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Figure 3 saved: {viz_path3}")

    # -- Console summary ------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Variant':<12} | {'F1≥0.5':<7} | {'Contains':<9} | {'Faith':<7} | "
          f"{'Ground':<7} | {'Quality':<8} | {'Refusal':<8} | {'Length':<6}")
    print("-" * 85)

    for variant in viz_variants:
        m = viz_aggregated[variant]
        print(f"{variant.capitalize():<12} | "
              f"{m['accuracy_f1']:>6.1%} | "
              f"{m['accuracy_contains']:>8.1%} | "
              f"{(m.get('avg_faithfulness') or 0):>6.2f} | "
              f"{(m.get('avg_groundedness') or 0):>6.2f} | "
              f"{(m.get('avg_quality_score') or 0):>7.2f} | "
              f"{m['refusal_rate']:>7.1%} | "
              f"{m['avg_answer_length']:>5.1f}w")

    print(f"\n{'Retrieval hit rate:':<25} {viz_avg_hit:.1%}")
    print(f"{'Questions evaluated:':<25} {viz_n:,}")
    if viz_metadata.get('total_cost'):
        print(f"{'Total cost:':<25} ${viz_metadata['total_cost']:.4f}")
        print(f"{'Cost per question:':<25} ${viz_metadata['total_cost'] / viz_n:.4f}")
    print(f"{'Faithfulness method:':<25} {viz_metadata.get('faithfulness_method', 'N/A')}")
    if viz_cost_meta:
        cascade_rate = viz_cost_meta.get('cascade_trigger_rate')
        agree_rate   = viz_cost_meta.get('cascade_agreement_rate')
        if cascade_rate is not None:
            print(f"{'Cascade trigger rate:':<25} {cascade_rate:.1%}  "
                  f"(target ~20% — {'⚠️ elevated' if cascade_rate > 0.40 else '✅ on target'})")
            if agree_rate is not None:
                print(f"{'Cascade agreement rate:':<25} {agree_rate:.1%}")
    print("=" * 80 + "\n")

    print(f"  Figures saved:")
    print(f"  1. {viz_path1}")
    print(f"  2. {viz_path2}")
    print(f"  3. {viz_path3}")

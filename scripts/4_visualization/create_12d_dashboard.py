#!/usr/bin/env python3
"""
Visualization System for Precision-Scored Methods
Creates comprehensive visualizations for methods scored across 12 dimensions

Generates 7 interactive visualizations:
- 4 2D plots: Impact vs Implementation, Scope vs Temporality, Time to Value vs Impact, People vs Process
- 3 3D plots: Strategic Cube, People×Process×Purpose Space, Adoption Space
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import argparse
from pathlib import Path

from cluster_utils import load_cluster_mappings


class MethodVisualizationDashboard:
    """
    Create comprehensive visualizations for scored methods
    """

    def __init__(self, df: pd.DataFrame, output_path: str = 'results/'):
        self.df = df
        self.output_path = output_path
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for visualization"""
        # Ensure we have the key columns
        if 'ease_score' not in self.df.columns and 'implementation_difficulty' in self.df.columns:
            self.df['ease_score'] = 100 - self.df['implementation_difficulty']
        if 'speed_score' not in self.df.columns and 'time_to_value' in self.df.columns:
            self.df['speed_score'] = self.df['time_to_value']

        # Add ROI and strategic scores if not present
        if 'score_roi' not in self.df.columns:
            self.df['score_roi'] = (
                self.df['impact_potential'] * 0.4 +
                self.df.get('ease_score', 50) * 0.3 +
                self.df.get('speed_score', 50) * 0.2 +
                self.df['applicability'] * 0.1
            )

        if 'score_strategic' not in self.df.columns:
            self.df['score_strategic'] = (
                self.df['impact_potential'] * 0.45 +
                self.df['applicability'] * 0.35 +
                self.df['scope'] * 0.2
            )

        if 'score_quick_wins' not in self.df.columns:
            self.df['score_quick_wins'] = (
                self.df.get('speed_score', 50) * 0.35 +
                self.df.get('ease_score', 50) * 0.35 +
                self.df['impact_potential'] * 0.2 +
                self.df['applicability'] * 0.1
            )

        if 'score_composite' not in self.df.columns:
            self.df['score_composite'] = (
                self.df['impact_potential'] * 0.25 +
                self.df['applicability'] * 0.2 +
                self.df.get('ease_score', 50) * 0.2 +
                self.df.get('speed_score', 50) * 0.15 +
                self.df['scope'] * 0.1 +
                self.df['temporality'] * 0.1
            )

    def create_comprehensive_dashboard(self):
        """Create all visualizations"""

        print("Creating comprehensive visualization dashboard...")

        # Create static matplotlib dashboard
        print("  1/3 Creating static dashboard...")
        self._create_static_dashboard()

        # Create interactive plotly dashboard
        print("  2/3 Creating interactive dashboard...")
        self._create_interactive_dashboard()

        # Create subcriteria analysis if available
        print("  3/3 Creating subcriteria analysis...")
        subcrit_cols = [col for col in self.df.columns if 'ease_adoption' in col or
                       'resources_required' in col or 'technical_complexity' in col or
                       'change_management' in col]
        if subcrit_cols:
            self._create_subcriteria_analysis()

        print("\n✅ All visualizations created successfully!")

    def _create_static_dashboard(self):
        """Create static visualization dashboard"""

        fig = plt.figure(figsize=(24, 20))

        # 1. Classic 2x2: Impact vs Ease
        ax1 = plt.subplot(3, 4, 1)
        self._plot_impact_vs_ease(ax1)

        # 2. Time to Value Distribution
        ax2 = plt.subplot(3, 4, 2)
        self._plot_time_to_value_distribution(ax2)

        # 3. Applicability Analysis
        ax3 = plt.subplot(3, 4, 3)
        self._plot_applicability_analysis(ax3)

        # 4. Category Distribution
        ax4 = plt.subplot(3, 4, 4)
        self._plot_category_distribution(ax4)

        # 5. Portfolio Matrix (BCG-style)
        ax5 = plt.subplot(3, 4, 5)
        self._plot_portfolio_matrix(ax5)

        # 6. Grade Distribution
        ax6 = plt.subplot(3, 4, 6)
        self._plot_grade_distribution(ax6)

        # 7. Correlation Heatmap
        ax7 = plt.subplot(3, 4, 7)
        self._plot_correlation_heatmap(ax7)

        # 8. Top Methods Comparison
        ax8 = plt.subplot(3, 4, 8, projection='polar')
        self._plot_top_methods_radar(ax8)

        # 9. Score Distributions
        ax9 = plt.subplot(3, 4, 9)
        self._plot_score_distributions(ax9)

        # 10. Implementation Roadmap
        ax10 = plt.subplot(3, 4, 10)
        self._plot_implementation_roadmap(ax10)

        # 11. Quick Wins vs Strategic
        ax11 = plt.subplot(3, 4, 11)
        self._plot_quick_vs_strategic(ax11)

        # 12. 3D Visualization
        ax12 = plt.subplot(3, 4, 12, projection='3d')
        self._plot_3d_space(ax12)

        plt.suptitle('Method Evaluation Dashboard - 9D Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        output_file = f'{self.output_path}evaluation_dashboard.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    ✓ Static dashboard saved to {output_file}")
        plt.close()

    def _plot_impact_vs_ease(self, ax):
        """Impact vs Implementation Ease matrix"""

        x = self.df['ease_score']
        y = self.df['impact_potential']

        # Color by time to value, size by applicability
        colors = self.df['speed_score']
        sizes = self.df['applicability'] * 3

        scatter = ax.scatter(x, y, c=colors, s=sizes, cmap='RdYlGn',
                           alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add quadrant lines
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=2)

        # Quadrant labels
        quadrants = [
            (25, 75, 'Difficult but\nImpactful', 'yellow'),
            (75, 75, 'Easy &\nImpactful', 'lightgreen'),
            (25, 25, 'Difficult &\nLow Impact', 'lightcoral'),
            (75, 25, 'Easy but\nLow Impact', 'lightgray')
        ]

        for x_pos, y_pos, label, color in quadrants:
            ax.text(x_pos, y_pos, label, ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3))

        ax.set_xlabel('Ease of Implementation →', fontsize=12, fontweight='bold')
        ax.set_ylabel('Impact Potential →', fontsize=12, fontweight='bold')
        ax.set_title('Impact vs Ease Matrix', fontsize=14, fontweight='bold')
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Speed to Value', rotation=270, labelpad=15)

    def _plot_portfolio_matrix(self, ax):
        """BCG-style portfolio matrix"""

        x = self.df['score_roi']
        y = self.df['score_strategic']

        # Determine portfolio categories
        categories = []
        colors = []
        for roi, strategic in zip(x, y):
            if roi >= 70 and strategic >= 70:
                categories.append('Stars')
                colors.append('gold')
            elif roi >= 70 and strategic < 70:
                categories.append('Cash Cows')
                colors.append('green')
            elif roi < 70 and strategic >= 70:
                categories.append('Question Marks')
                colors.append('orange')
            else:
                categories.append('Dogs')
                colors.append('red')

        ax.scatter(x, y, c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=1)

        # Add quadrant lines
        ax.axhline(y=70, color='black', linestyle='-', linewidth=2)
        ax.axvline(x=70, color='black', linestyle='-', linewidth=2)

        # Labels
        ax.text(85, 85, 'STARS', ha='center', fontsize=12, fontweight='bold')
        ax.text(85, 35, 'CASH COWS', ha='center', fontsize=12, fontweight='bold')
        ax.text(35, 85, 'QUESTIONS', ha='center', fontsize=12, fontweight='bold')
        ax.text(35, 35, 'DOGS', ha='center', fontsize=12, fontweight='bold')

        ax.set_xlabel('ROI Score →', fontsize=12, fontweight='bold')
        ax.set_ylabel('Strategic Value →', fontsize=12, fontweight='bold')
        ax.set_title('Strategic Portfolio Matrix', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2)

    def _plot_time_to_value_distribution(self, ax):
        """Distribution of time to value"""

        # Create bins for time categories
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['Immediate\n(days)', 'Short\n(months)', 'Medium\n(quarters)',
                 'Long\n(1-2 years)', 'Very Long\n(2+ years)']

        self.df['time_category'] = pd.cut(self.df['time_to_value'], bins=bins, labels=labels)

        # Count and plot
        counts = self.df['time_category'].value_counts().sort_index()
        colors = ['darkgreen', 'green', 'yellow', 'orange', 'red']

        bars = ax.bar(range(len(counts)), counts.values, color=colors, alpha=0.7, edgecolor='black')

        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, fontsize=9)
        ax.set_ylabel('Number of Methods', fontsize=11)
        ax.set_title('Time to Value Distribution', fontsize=14, fontweight='bold')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)

        ax.grid(True, axis='y', alpha=0.3)

    def _plot_applicability_analysis(self, ax):
        """Applicability distribution and analysis"""

        # Create violin plot
        parts = ax.violinplot([self.df['applicability']], positions=[0.5],
                             showmeans=True, showmedians=True)

        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

        # Add percentile lines
        percentiles = [25, 50, 75]
        for p in percentiles:
            val = np.percentile(self.df['applicability'], p)
            ax.axhline(y=val, color='gray', linestyle='--', alpha=0.5)
            ax.text(1.1, val, f'{p}th %ile: {val:.0f}', va='center', fontsize=9)

        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 100)
        ax.set_xticks([0.5])
        ax.set_xticklabels(['All Methods'])
        ax.set_ylabel('Applicability Score', fontsize=11)
        ax.set_title('Applicability Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    def _plot_category_distribution(self, ax):
        """Pie chart of method categories"""

        if 'category' not in self.df.columns:
            ax.text(0.5, 0.5, 'Categories not available', ha='center', va='center')
            return

        category_counts = self.df['category'].value_counts()

        # Map emoji categories to clean names
        category_map = {
            '⭐ Stars': 'Stars',
            '⚡ Quick Wins': 'Quick Wins',
            '🚀 Strategic Investments': 'Strategic Investments',
            '🏗️ Foundation Builders': 'Foundation Builders',
            '💎 High-Risk High-Reward': 'High-Risk High-Reward',
            '🎯 Niche Solutions': 'Niche Solutions',
            '🐢 Long-Term Investments': 'Long-Term Investments',
            '📊 Standard Practices': 'Standard Practices'
        }

        colors = {
            'Stars': 'gold',
            'Quick Wins': 'lightgreen',
            'Strategic Investments': 'royalblue',
            'Foundation Builders': 'brown',
            'High-Risk High-Reward': 'purple',
            'Niche Solutions': 'gray',
            'Long-Term Investments': 'orange',
            'Standard Practices': 'lightgray'
        }

        # Clean labels and colors
        clean_labels = [category_map.get(cat, cat) for cat in category_counts.index]
        wedge_colors = [colors.get(category_map.get(cat, cat), 'gray') for cat in category_counts.index]

        wedges, texts, autotexts = ax.pie(category_counts.values,
                                          labels=clean_labels,
                                          colors=wedge_colors,
                                          autopct='%1.1f%%',
                                          startangle=90)

        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('black')
            autotext.set_weight('bold')

        ax.set_title('Method Categories', fontsize=14, fontweight='bold')

    def _plot_correlation_heatmap(self, ax):
        """Correlation between dimensions and scores"""

        corr_cols = ['scope', 'temporality', 'ease_adoption', 'resources_required',
                    'technical_complexity', 'change_management_difficulty',
                    'implementation_difficulty', 'impact_potential',
                    'time_to_value', 'applicability']

        # Filter to available columns
        corr_cols = [col for col in corr_cols if col in self.df.columns]

        if len(corr_cols) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center')
            return

        corr_matrix = self.df[corr_cols].corr()

        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

        # Labels
        labels = [col.replace('_', '\n') for col in corr_cols]
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        # Add correlation values
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', fontsize=7, color=text_color)

        ax.set_title('Dimension Correlations', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)

    def _plot_top_methods_radar(self, ax):
        """Radar chart of top methods"""

        # Get top 5 methods
        top_5 = self.df.nlargest(5, 'score_composite')

        categories = ['Ease', 'Impact', 'Speed', 'Applicability']
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        for idx, (_, method) in enumerate(top_5.iterrows()):
            values = [
                method.get('ease_score', 100 - method.get('implementation_difficulty', 50)),
                method['impact_potential'],
                method.get('speed_score', method.get('time_to_value', 50)),
                method['applicability']
            ]
            values += values[:1]

            label = method['name'][:20] + '...' if len(method['name']) > 20 else method['name']
            ax.plot(angles, values, 'o-', linewidth=2, label=label, alpha=0.7)
            ax.fill(angles, values, alpha=0.15)

        ax.set_ylim(0, 100)
        ax.set_title('Top 5 Methods Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.grid(True)

    def _plot_3d_space(self, ax):
        """3D visualization of method space"""

        x = self.df.get('ease_score', 100 - self.df.get('implementation_difficulty', 50))
        y = self.df['impact_potential']
        z = self.df['applicability']
        colors = self.df.get('speed_score', self.df.get('time_to_value', 50))

        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis',
                           s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Ease', fontsize=10)
        ax.set_ylabel('Impact', fontsize=10)
        ax.set_zlabel('Applicability', fontsize=10)
        ax.set_title('3D Method Space', fontsize=14, fontweight='bold')

        plt.colorbar(scatter, ax=ax, label='Speed', shrink=0.8, pad=0.1)

    def _plot_grade_distribution(self, ax):
        """Grade distribution histogram"""

        if 'letter_grade' in self.df.columns:
            grade_col = 'letter_grade'
        elif 'grade' in self.df.columns:
            grade_col = 'grade'
        else:
            # Create grades from composite score
            grades = []
            for score in self.df['score_composite']:
                if score >= 90: grades.append('A+')
                elif score >= 85: grades.append('A')
                elif score >= 80: grades.append('A-')
                elif score >= 75: grades.append('B+')
                elif score >= 70: grades.append('B')
                elif score >= 65: grades.append('B-')
                elif score >= 60: grades.append('C+')
                elif score >= 55: grades.append('C')
                elif score >= 50: grades.append('C-')
                elif score >= 45: grades.append('D+')
                elif score >= 40: grades.append('D')
                else: grades.append('F')
            self.df['grade'] = grades
            grade_col = 'grade'

        grade_order = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F']
        grade_counts = self.df[grade_col].value_counts()

        # Ensure all grades are represented
        for grade in grade_order:
            if grade not in grade_counts:
                grade_counts[grade] = 0

        grade_counts = grade_counts.reindex(grade_order, fill_value=0)

        colors = ['darkgreen' if g.startswith('A') else
                 'green' if g.startswith('B') else
                 'yellow' if g.startswith('C') else
                 'orange' if g.startswith('D') else 'red'
                 for g in grade_order]

        bars = ax.bar(range(len(grade_order)), grade_counts.values,
                      color=colors, alpha=0.7, edgecolor='black')

        ax.set_xticks(range(len(grade_order)))
        ax.set_xticklabels(grade_order, fontsize=9)
        ax.set_ylabel('Number of Methods', fontsize=11)
        ax.set_title('Grade Distribution', fontsize=14, fontweight='bold')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)

        ax.grid(True, axis='y', alpha=0.3)

    def _plot_score_distributions(self, ax):
        """Distribution of all scores"""

        scores_to_plot = ['score_roi', 'score_quick_wins', 'score_strategic', 'score_composite']
        scores_available = [s for s in scores_to_plot if s in self.df.columns]

        if not scores_available:
            ax.text(0.5, 0.5, 'Scores not available', ha='center', va='center')
            return

        data = [self.df[score].values for score in scores_available]
        labels = [score.replace('score_', '').replace('_', ' ').title()
                 for score in scores_available]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Score Distributions', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 100)

    def _plot_implementation_roadmap(self, ax):
        """Suggested implementation roadmap"""

        # Create phases based on scores
        phases = {
            'Phase 1: Quick Wins': self.df[
                (self.df.get('speed_score', 50) >= 70) &
                (self.df.get('ease_score', 50) >= 70)
            ].nlargest(5, 'score_composite' if 'score_composite' in self.df.columns else 'impact_potential'),

            'Phase 2: Foundation': self.df[
                (self.df['applicability'] >= 70) &
                (self.df.get('ease_score', 50) >= 60)
            ].nlargest(5, 'score_composite' if 'score_composite' in self.df.columns else 'impact_potential'),

            'Phase 3: Strategic': self.df[
                (self.df['impact_potential'] >= 70)
            ].nlargest(5, 'score_strategic' if 'score_strategic' in self.df.columns else 'impact_potential')
        }

        y_pos = 0.9
        colors = ['green', 'blue', 'purple']

        for (phase_name, phase_df), color in zip(phases.items(), colors):
            ax.text(0.1, y_pos, phase_name, fontsize=12, fontweight='bold', color=color)
            y_pos -= 0.08

            for _, method in phase_df.iterrows():
                method_text = method['name'][:40] + '...' if len(method['name']) > 40 else method['name']
                score_text = f" ({method.get('score_composite', method['impact_potential']):.0f})"
                ax.text(0.15, y_pos, f"• {method_text}{score_text}", fontsize=9)
                y_pos -= 0.06

            y_pos -= 0.04

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Implementation Roadmap', fontsize=14, fontweight='bold')

    def _plot_quick_vs_strategic(self, ax):
        """Quick wins vs strategic value comparison"""

        x = self.df['score_quick_wins']
        y = self.df['score_strategic']

        scatter = ax.scatter(x, y, c=self.df['impact_potential'],
                           s=50, cmap='YlOrRd', alpha=0.6,
                           edgecolors='black', linewidth=0.5)

        # Add diagonal line
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Balanced')

        ax.set_xlabel('Quick Win Potential →', fontsize=12, fontweight='bold')
        ax.set_ylabel('Strategic Value →', fontsize=12, fontweight='bold')
        ax.set_title('Quick Wins vs Strategic Value', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.colorbar(scatter, ax=ax, label='Impact')

    def _create_portfolio_matrix_interactive(self, category_info):
        """Create interactive BCG-style portfolio matrix (Stars, Cash Cows, Question Marks, Dogs)"""

        fig = go.Figure()

        # Calculate ROI and Strategic scores if not present
        x = self.df['score_roi']
        y = self.df['score_strategic']

        # Determine BCG category for each method
        bcg_categories = []
        bcg_colors = []
        for roi, strategic in zip(x, y):
            if roi >= 70 and strategic >= 70:
                bcg_categories.append('Stars')
                bcg_colors.append('gold')
            elif roi >= 70 and strategic < 70:
                bcg_categories.append('Cash Cows')
                bcg_colors.append('green')
            elif roi < 70 and strategic >= 70:
                bcg_categories.append('Question Marks')
                bcg_colors.append('orange')
            else:
                bcg_categories.append('Dogs')
                bcg_colors.append('red')

        self.df['bcg_category'] = bcg_categories

        # BCG category colors and info
        bcg_info = {
            'Stars': {'color': '#FFD700', 'symbol': 'star'},
            'Cash Cows': {'color': '#228B22', 'symbol': 'circle'},
            'Question Marks': {'color': '#FF8C00', 'symbol': 'diamond'},
            'Dogs': {'color': '#DC143C', 'symbol': 'x'}
        }

        categories_sorted = sorted(self.df['method_category'].unique())

        # Store all point data for click annotations
        all_points_data = []
        point_index = 0

        # Add traces for each semantic category (colored by semantic category)
        for category in categories_sorted:
            cat_df = self.df[self.df['method_category'] == category]

            if len(cat_df) == 0:
                continue

            cat_info = category_info.get(category, {'color': '#95a5a6', 'name': 'Unknown'})
            cat_color = cat_info['color']
            cat_display_name = cat_info['name']

            # Build hover text and store point metadata
            hover_texts = []
            for _, method in cat_df.iterrows():
                hover = (
                    f"<b>{method['name']}</b><br>" +
                    f"Semantic Category: {cat_display_name}<br>" +
                    f"BCG Quadrant: {method['bcg_category']}<br>" +
                    f"Source: {method.get('source', 'N/A')}<br>" +
                    f"ROI Score: {method['score_roi']:.1f}/100<br>" +
                    f"Strategic Value: {method['score_strategic']:.1f}/100<br>" +
                    f"Impact: {method.get('impact_potential', 0):.1f}/100<br>" +
                    f"Ease: {method.get('ease_score', 0):.1f}/100"
                )
                hover_texts.append(hover)

                # Store metadata for annotations
                all_points_data.append({
                    'x': method['score_roi'],
                    'y': method['score_strategic'],
                    'name': method['name'],
                    'category': cat_display_name,
                    'bcg_category': method['bcg_category'],
                    'source': method.get('source', 'N/A'),
                    'roi': method['score_roi'],
                    'strategic': method['score_strategic'],
                    'color': cat_color,
                    'index': point_index
                })
                point_index += 1

            fig.add_trace(
                go.Scatter(
                    x=cat_df['score_roi'],
                    y=cat_df['score_strategic'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=cat_color,
                        line=dict(color='black', width=0.5),
                        opacity=0.8
                    ),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    name=cat_display_name,
                    visible=True,
                    customdata=[[d['index']] for d in all_points_data[-len(cat_df):]]
                )
            )

        # Store point data in the figure for JavaScript access
        fig._points_data = all_points_data

        # Add quadrant lines at 70 (threshold for BCG matrix)
        fig.add_hline(y=70, line_dash="solid", line_color="black", line_width=2, opacity=0.7)
        fig.add_vline(x=70, line_dash="solid", line_color="black", line_width=2, opacity=0.7)

        # Add quadrant shading and labels
        # Stars (top-right)
        fig.add_shape(type="rect", x0=70, y0=70, x1=100, y1=100,
                     fillcolor="rgba(255, 215, 0, 0.1)", line=dict(width=0),
                     layer='below')  # Render below data points
        fig.add_annotation(x=85, y=92, text="<b>⭐ STARS</b><br>High ROI, High Strategic",
                          showarrow=False, font=dict(size=12, color="darkgoldenrod"),
                          bgcolor="rgba(255,255,255,0.7)")

        # Cash Cows (bottom-right)
        fig.add_shape(type="rect", x0=70, y0=0, x1=100, y1=70,
                     fillcolor="rgba(34, 139, 34, 0.1)", line=dict(width=0),
                     layer='below')  # Render below data points
        fig.add_annotation(x=85, y=10, text="<b>🐄 CASH COWS</b><br>High ROI, Low Strategic",
                          showarrow=False, font=dict(size=12, color="darkgreen"),
                          bgcolor="rgba(255,255,255,0.7)")

        # Question Marks (top-left)
        fig.add_shape(type="rect", x0=0, y0=70, x1=70, y1=100,
                     fillcolor="rgba(255, 140, 0, 0.1)", line=dict(width=0),
                     layer='below')  # Render below data points
        fig.add_annotation(x=35, y=92, text="<b>❓ QUESTION MARKS</b><br>Low ROI, High Strategic",
                          showarrow=False, font=dict(size=12, color="darkorange"),
                          bgcolor="rgba(255,255,255,0.7)")

        # Dogs (bottom-left)
        fig.add_shape(type="rect", x0=0, y0=0, x1=70, y1=70,
                     fillcolor="rgba(220, 20, 60, 0.1)", line=dict(width=0),
                     layer='below')  # Render below data points
        fig.add_annotation(x=35, y=10, text="<b>🐕 DOGS</b><br>Low ROI, Low Strategic",
                          showarrow=False, font=dict(size=12, color="crimson"),
                          bgcolor="rgba(255,255,255,0.7)")

        # Create dropdown buttons for filtering by semantic category
        buttons = [
            dict(
                label="Show All Categories",
                method="update",
                args=[{"visible": [True] * len(categories_sorted)}]
            )
        ]

        for idx, category in enumerate(categories_sorted):
            cat_df = self.df[self.df['method_category'] == category]
            cat_display_name = category_info.get(category, {'name': 'Unknown'})['name']
            visible_list = [False] * len(categories_sorted)
            visible_list[idx] = True

            buttons.append(dict(
                label=f"{cat_display_name} ({len(cat_df)} methods)",
                method="update",
                args=[{"visible": visible_list}]
            ))

        # Add buttons to filter by BCG quadrant
        buttons.append(dict(label="─── By BCG Quadrant ───", method="skip", args=[{}]))

        for bcg_cat in ['Stars', 'Cash Cows', 'Question Marks', 'Dogs']:
            bcg_count = len(self.df[self.df['bcg_category'] == bcg_cat])
            # For BCG filtering, we show all traces but will use a different approach
            buttons.append(dict(
                label=f"{bcg_cat} ({bcg_count} methods)",
                method="update",
                args=[{"visible": [True] * len(categories_sorted)}]  # Show all, hover will show BCG
            ))

        # Update layout
        fig.update_layout(
            title=dict(
                text='Strategic Portfolio Matrix (BCG Style)<br><sup>ROI Score vs Strategic Value - Threshold at 70</sup>',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='← Low ROI | ROI Score | High ROI →',
                range=[-5, 105],
                showgrid=True,
                gridcolor='lightgray',
                dtick=10
            ),
            yaxis=dict(
                title='← Low Strategic | Strategic Value | High Strategic →',
                range=[-5, 105],
                showgrid=True,
                gridcolor='lightgray',
                dtick=10
            ),
            hovermode='closest',
            width=1400,
            height=1000,
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                )
            ]
        )

        return fig

    def _create_single_3d_plot(self, plot_config, category_info):
        """Create a single independent 3D plot with its own legend and dropdown"""

        fig = go.Figure()

        categories_sorted = sorted(self.df['method_category'].unique())

        # Add traces for each category
        for category in categories_sorted:
            cat_df = self.df[self.df['method_category'] == category]

            if len(cat_df) == 0:
                continue

            cat_info = category_info.get(category, {'color': '#95a5a6', 'name': 'Unknown'})
            cat_color = cat_info['color']
            cat_display_name = cat_info['name']

            # Build hover text
            hover_texts = []
            for _, method in cat_df.iterrows():
                hover = (
                    f"<b>{method['name']}</b><br>" +
                    f"Category: {cat_display_name}<br>" +
                    f"Source: {method.get('source', 'N/A')}<br>" +
                    f"{plot_config.get('x_title', plot_config['x']).replace('_', ' ').title()}: {method.get(plot_config['x'], 0):.1f}/100<br>" +
                    f"{plot_config.get('y_title', plot_config['y']).replace('_', ' ').title()}: {method.get(plot_config['y'], 0):.1f}/100<br>" +
                    f"{plot_config.get('z_title', plot_config['z']).replace('_', ' ').title()}: {method.get(plot_config['z'], 0):.1f}/100"
                )
                hover_texts.append(hover)

            fig.add_trace(
                go.Scatter3d(
                    x=cat_df[plot_config['x']],
                    y=cat_df[plot_config['y']],
                    z=cat_df[plot_config['z']],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=cat_color,
                        line=dict(color='black', width=0.5),
                        opacity=0.8
                    ),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    name=cat_display_name,
                    visible=True
                )
            )

        # Create dropdown buttons
        buttons = [
            dict(
                label="Show All Categories",
                method="update",
                args=[{"visible": [True] * len(categories_sorted)}]
            )
        ]

        for idx, category in enumerate(categories_sorted):
            cat_df = self.df[self.df['method_category'] == category]
            cat_display_name = category_info.get(category, {'name': 'Unknown'})['name']
            visible_list = [False] * len(categories_sorted)
            visible_list[idx] = True

            buttons.append(dict(
                label=f"{cat_display_name} ({len(cat_df)} methods)",
                method="update",
                args=[{"visible": visible_list}]
            ))

        # Update layout with tick labels that rotate with the plot
        fig.update_layout(
            title=dict(
                text=f'{plot_config["title"]}<br>',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text=plot_config.get('x_title', plot_config['x']), font=dict(size=14)),
                    showgrid=True,
                    gridcolor='lightgray',
                    tickmode='array',
                    tickvals=plot_config.get('x_tickvals', [0, 20, 40, 60, 80, 100]),
                    ticktext=plot_config.get('x_ticktext', ['0', '20', '40', '60', '80', '100']),
                    showticklabels=True
                ),
                yaxis=dict(
                    title=dict(text=plot_config.get('y_title', plot_config['y']), font=dict(size=14)),
                    showgrid=True,
                    gridcolor='lightgray',
                    tickmode='array',
                    tickvals=plot_config.get('y_tickvals', [0, 20, 40, 60, 80, 100]),
                    ticktext=plot_config.get('y_ticktext', ['0', '20', '40', '60', '80', '100']),
                    showticklabels=True
                ),
                zaxis=dict(
                    title=dict(text=plot_config.get('z_title', plot_config['z']), font=dict(size=14)),
                    showgrid=True,
                    gridcolor='lightgray',
                    tickmode='array',
                    tickvals=plot_config.get('z_tickvals', [0, 20, 40, 60, 80, 100]),
                    ticktext=plot_config.get('z_ticktext', ['0', '20', '40', '60', '80', '100']),
                    showticklabels=True
                )
            ),
            hovermode='closest',
            width=1400,
            height=1000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                )
            ]
        )

        return fig

    def _create_single_plot(self, plot_config, category_info):
        """Create a single independent plot with its own legend and dropdown"""

        fig = go.Figure()

        categories_sorted = sorted(self.df['method_category'].unique())

        # Store all point data for click annotations
        all_points_data = []
        point_index = 0

        # Add traces for each category
        for category in categories_sorted:
            cat_df = self.df[self.df['method_category'] == category]

            if len(cat_df) == 0:
                continue

            cat_info = category_info.get(category, {'color': '#95a5a6', 'name': 'Unknown'})
            cat_color = cat_info['color']
            cat_display_name = cat_info['name']

            # Build hover text and store point metadata
            hover_texts = []
            for _, method in cat_df.iterrows():
                hover = (
                    f"<b>{method['name']}</b><br>" +
                    f"Category: {cat_display_name}<br>" +
                    f"Source: {method.get('source', 'N/A')}<br>" +
                    f"{plot_config['title'].split(' vs ')[0]}: {method.get(plot_config['x'], 0):.1f}/100<br>" +
                    f"{plot_config['title'].split(' vs ')[1]}: {method.get(plot_config['y'], 0):.1f}/100"
                )
                hover_texts.append(hover)

                # Store metadata for annotations
                all_points_data.append({
                    'x': method.get(plot_config['x'], 0),
                    'y': method.get(plot_config['y'], 0),
                    'name': method['name'],
                    'category': cat_display_name,
                    'source': method.get('source', 'N/A'),
                    'color': cat_color,
                    'index': point_index
                })
                point_index += 1

            fig.add_trace(
                go.Scatter(
                    x=cat_df[plot_config['x']],
                    y=cat_df[plot_config['y']],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=cat_color,
                        line=dict(color='black', width=0.5),
                        opacity=0.8
                    ),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    name=cat_display_name,
                    visible=True,
                    customdata=[[d['index']] for d in all_points_data[-len(cat_df):]]
                )
            )

        # Store point data in the figure for JavaScript access
        fig._points_data = all_points_data

        # Add quadrant lines at 50
        fig.add_hline(y=50, line_dash="dash", line_color="black", opacity=0.4)
        fig.add_vline(x=50, line_dash="dash", line_color="black", opacity=0.4)

        # Create dropdown buttons
        buttons = [
            dict(
                label="Show All Categories",
                method="update",
                args=[{"visible": [True] * len(categories_sorted)}]
            )
        ]

        for idx, category in enumerate(categories_sorted):
            cat_df = self.df[self.df['method_category'] == category]
            cat_display_name = category_info.get(category, {'name': 'Unknown'})['name']
            visible_list = [False] * len(categories_sorted)
            visible_list[idx] = True

            buttons.append(dict(
                label=f"{cat_display_name} ({len(cat_df)} methods)",
                method="update",
                args=[{"visible": visible_list}]
            ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{plot_config["title"]}<br>',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=plot_config['x_label'],
                range=[-5, 105],
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title=plot_config['y_label'],
                range=[-5, 105],
                showgrid=True,
                gridcolor='lightgray'
            ),
            hovermode='closest',
            width=1400,
            height=1000,
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                )
            ]
        )

        return fig

    def _create_interactive_dashboard(self):
        """Create interactive Plotly dashboard with completely separate plots"""

        # Load semantic clustering categories dynamically from dendrogram_categories.json
        clusters_path = Path('results_semantic_clustering_combined/combined_clusters.json')
        if not clusters_path.exists():
            print("    ⚠ Warning: combined_clusters.json not found, using placeholder categories")
            self.df['method_category'] = 'uncategorized'
            category_info = {'uncategorized': {'color': '#95a5a6', 'name': 'Uncategorized'}}
        else:
            # Load cluster mappings from cluster_utils (uses dendrogram_categories.json)
            cluster_mappings = load_cluster_mappings()

            with open(clusters_path, 'r') as f:
                clusters_data = json.load(f)

            # Build category lookup from method NAME to category key
            # (Using name because index values differ between scores and clustering files)
            method_name_to_category = {}
            for cluster_id, cluster_data in clusters_data['clusters'].items():
                # Get category key for this cluster from dynamic mappings
                category_key = cluster_mappings['cluster_to_synergy'].get(cluster_id, 'uncategorized')
                for method in cluster_data['methods']:
                    # Normalize method name for matching
                    method_name = method['Method'].strip().lower()
                    method_name_to_category[method_name] = category_key

            # Map to dataframe using method name
            self.df['method_category'] = self.df['name'].apply(
                lambda name: method_name_to_category.get(name.strip().lower() if isinstance(name, str) else '', 'uncategorized')
            )

        # Build category_info dynamically from cluster_mappings
        # Color palette - 20 distinct colors for categories
        color_palette = [
            '#E41A1C',  # Red
            '#377EB8',  # Blue
            '#4DAF4A',  # Green
            '#984EA3',  # Purple
            '#FF7F00',  # Orange
            '#FFFF33',  # Yellow
            '#A65628',  # Brown
            '#F781BF',  # Pink
            '#999999',  # Gray
            '#66C2A5',  # Teal
            '#FC8D62',  # Coral
            '#8DA0CB',  # Periwinkle
            '#E78AC3',  # Orchid
            '#A6D854',  # Lime
            '#FFD92F',  # Gold
            '#1B9E77',  # Dark Teal
            '#D95F02',  # Dark Orange
            '#7570B3',  # Slate
            '#E7298A',  # Magenta
            '#66A61E',  # Olive
        ]

        category_info = {}
        for i, (cat_key, display_name) in enumerate(cluster_mappings['synergy_display_names'].items()):
            category_info[cat_key] = {
                'color': color_palette[i % len(color_palette)],
                'name': display_name
            }
        # Add uncategorized fallback
        category_info['uncategorized'] = {'color': '#B3B3B3', 'name': 'Uncategorized'}

        # Debug: print unique categories found
        print(f"    Found {len(self.df['method_category'].unique())} unique categories:")
        for cat in sorted(self.df['method_category'].unique()):
            count = len(self.df[self.df['method_category'] == cat])
            color = category_info.get(cat, {'color': '#95a5a6'})['color']
            print(f"      - {cat}: {count} methods (color: {color})")

        # Create list of 2D plots with verified directional labels
        # Note: Scores range 0-100, with direction indicated by arrows
        plots_config = [
            {
                'x': 'implementation_difficulty',
                'y': 'impact_potential',
                'title': 'Impact vs Implementation Difficulty',
                'x_label': '← Easy | Implementation Difficulty | Hard →',
                'y_label': '← Low | Impact Potential | High →'
            },
            {
                'x': 'scope',
                'y': 'temporality',
                'title': 'Scope vs Temporality',
                'x_label': '← Tactical | Scope | Strategic →',
                'y_label': '← Immediate | Temporality | Evolutionary →'
            },
            {
                'x': 'time_to_value',
                'y': 'impact_potential',
                'title': 'Time to Value vs Impact',
                'x_label': '← Slow | Time to Value | Fast →',
                'y_label': '← Low | Impact Potential | High →'
            },
            {
                'x': 'people_focus',
                'y': 'process_focus',
                'title': 'People vs Process Focus',
                'x_label': '← Technical/System | People Focus | Human →',
                'y_label': '← Ad-hoc | Process Focus | Systematic →'
            }
        ]

        # 3D plots configuration with tick labels that rotate with the plot
        # Labels at BOTH ends (0 and 100) for visibility from any angle
        plots_3d_config = [
            {
                'x': 'scope',
                'y': 'impact_potential',
                'z': 'implementation_difficulty',
                'title': 'The Strategic Cube',
                'x_title': 'Scope: Tactical (0) ↔ Strategic (100)',
                'y_title': 'Impact: Low (0) ↔ High (100)',
                'z_title': 'Implementation: Easy (0) ↔ Hard (100)',
                'x_tickvals': [0, 20, 40, 60, 80, 100],
                'x_ticktext': ['0: Tactical', '20', '40', '60', '80', '100: Strategic'],
                'y_tickvals': [0, 20, 40, 60, 80, 100],
                'y_ticktext': ['0: Low', '20', '40', '60', '80', '100: High'],
                'z_tickvals': [0, 20, 40, 60, 80, 100],
                'z_ticktext': ['0: Easy', '20', '40', '60', '80', '100: Hard']
            },
            {
                'x': 'people_focus',
                'y': 'process_focus',
                'z': 'purpose_orientation',
                'title': 'People × Process × Purpose Space',
                'x_title': 'People: Technical/System (0) ↔ Human (100)',
                'y_title': 'Process: Ad-hoc (0) ↔ Systematic (100)',
                'z_title': 'Purpose: Internal (0) ↔ External (100)',
                'x_tickvals': [0, 20, 40, 60, 80, 100],
                'x_ticktext': ['0: Tech/Sys', '20', '40', '60', '80', '100: Human'],
                'y_tickvals': [0, 20, 40, 60, 80, 100],
                'y_ticktext': ['0: Ad-hoc', '20', '40', '60', '80', '100: Systematic'],
                'z_tickvals': [0, 20, 40, 60, 80, 100],
                'z_ticktext': ['0: Internal', '20', '40', '60', '80', '100: External']
            },
            {
                'x': 'ease_adoption',
                'y': 'change_management_difficulty',
                'z': 'time_to_value',
                'title': 'The Adoption Space',
                'x_title': 'Ease of Adoption: Hard (0) ↔ Easy (100)',
                'y_title': 'Change Mgmt: Easy (0) ↔ Hard (100)',
                'z_title': 'Time to Value: Slow (0) ↔ Fast (100)',
                'x_tickvals': [0, 20, 40, 60, 80, 100],
                'x_ticktext': ['0: Hard', '20', '40', '60', '80', '100: Easy'],
                'y_tickvals': [0, 20, 40, 60, 80, 100],
                'y_ticktext': ['0: Easy', '20', '40', '60', '80', '100: Hard'],
                'z_tickvals': [0, 20, 40, 60, 80, 100],
                'z_ticktext': ['0: Slow', '20', '40', '60', '80', '100: Fast']
            }
        ]

        # Create 2D plots (4 plots)
        plot_htmls = []
        plot_data_json = []

        for idx, plot_config in enumerate(plots_config, 1):
            print(f"    Creating 2D plot {idx}/{len(plots_config)}: {plot_config['title']}")
            fig = self._create_single_plot(plot_config, category_info)

            # Store point data for JavaScript
            if hasattr(fig, '_points_data'):
                plot_data_json.append(json.dumps(fig._points_data))
            else:
                plot_data_json.append('[]')

            # Convert to HTML div with editable config
            config = {
                'editable': True,
                'edits': {
                    'annotationPosition': True,
                    'annotationTail': True,
                    'annotationText': False
                }
            }
            plot_html = fig.to_html(
                include_plotlyjs=(idx == 1),
                full_html=False,
                div_id=f'plot2d_{idx}',
                config=config
            )
            plot_htmls.append(plot_html)

        # Create BCG Portfolio Matrix plot
        print(f"    Creating BCG Portfolio Matrix...")
        portfolio_fig = self._create_portfolio_matrix_interactive(category_info)

        # Store BCG portfolio point data for JavaScript
        if hasattr(portfolio_fig, '_points_data'):
            portfolio_data_json = json.dumps(portfolio_fig._points_data)
        else:
            portfolio_data_json = '[]'

        # Use editable config for BCG matrix too
        config = {
            'editable': True,
            'edits': {
                'annotationPosition': True,
                'annotationTail': True,
                'annotationText': False
            }
        }
        portfolio_html = portfolio_fig.to_html(
            include_plotlyjs=False,
            full_html=False,
            div_id='plot_portfolio',
            config=config
        )
        plot_htmls.append(portfolio_html)

        # Create 3D plots (3 plots)
        for idx, plot_config in enumerate(plots_3d_config, 1):
            print(f"    Creating 3D plot {idx}/{len(plots_3d_config)}: {plot_config['title']}")
            fig = self._create_single_3d_plot(plot_config, category_info)

            # Convert to HTML div (not full page)
            plot_html = fig.to_html(include_plotlyjs=False, full_html=False, div_id=f'plot3d_{idx}')
            plot_htmls.append(plot_html)

        # Combine into single HTML page
        output_file = f"{self.output_path}interactive_dashboard.html"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>12D Interactive Dashboard - {len(self.df)} Methods</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .plot-container {{
            background-color: white;
            margin-bottom: 30px;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
        }}
        .section-header {{
            background-color: #2c3e50;
            color: white;
            padding: 15px;
            margin-top: 40px;
            margin-bottom: 20px;
            border-radius: 5px;
            font-size: 24px;
            font-weight: bold;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .info-banner {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 12px 15px;
            margin: 10px 0 20px 0;
            border-radius: 4px;
            font-size: 14px;
            color: #2c3e50;
        }}
        .info-banner strong {{
            color: #2980b9;
        }}
    </style>
</head>
<body>
    <h1>Method analysis result visualizations - {len(self.df)} Methods</h1>

    <div class="info-banner">
        <strong>💡 Interactive Features:</strong>
        Click any data point to pin/unpin a label with details.
        Pinned labels are draggable - click and drag to reposition them and avoid overlaps.
        Click the point again to remove the label.
    </div>

    <div class="section-header"> </div>

    <div class="plot-container">
        <h2>1. Impact vs Implementation Difficulty</h2>
        {plot_htmls[0]}
    </div>

    <div class="plot-container">
        <h2>2. Scope vs Temporality</h2>
        {plot_htmls[1]}
    </div>

    <div class="plot-container">
        <h2>3. Time to Value vs Impact</h2>
        {plot_htmls[2]}
    </div>

    <div class="plot-container">
        <h2>4. People vs Process Focus</h2>
        {plot_htmls[3]}
    </div>

    <div class="section-header">📊 Strategic Analysis</div>

    <div class="plot-container">
        <h2>5. Strategic Portfolio Matrix (BCG Style)</h2>
        <p style="color: #666; margin-left: 10px;">Stars, Cash Cows, Question Marks, Dogs - based on ROI Score vs Strategic Value (threshold: 70)</p>
        {plot_htmls[4]}
    </div>

    <div class="section-header">🎲 3D Visualizations</div>

    <div class="plot-container">
        <h2>6. The Strategic Cube (Scope × Impact × Implementation)</h2>
        {plot_htmls[5]}
    </div>

    <div class="plot-container">
        <h2>7. People × Process × Purpose Space</h2>
        {plot_htmls[6]}
    </div>

    <div class="plot-container">
        <h2>8. The Adoption Space (Ease × Change Mgmt × Time to Value)</h2>
        {plot_htmls[7]}
    </div>

    <script>
        // Point data for each 2D plot (for click-to-pin annotations)
        const plotsData = [
            {plot_data_json[0]},
            {plot_data_json[1]},
            {plot_data_json[2]},
            {plot_data_json[3]}
        ];

        // Point data for BCG Portfolio Matrix
        const portfolioData = {portfolio_data_json};

        // Track active annotations per plot
        const activeAnnotations = {{}};

        // Add click handlers to 2D plots
        ['plot2d_1', 'plot2d_2', 'plot2d_3', 'plot2d_4'].forEach((plotId, plotIdx) => {{
            const plotDiv = document.getElementById(plotId);
            if (!plotDiv) return;

            activeAnnotations[plotId] = new Set();

            plotDiv.on('plotly_click', function(data) {{
                const point = data.points[0];
                if (!point) return;

                // Get point metadata
                const pointIndex = point.customdata ? point.customdata[0] : point.pointIndex;
                const pointData = plotsData[plotIdx][pointIndex];
                if (!pointData) return;

                // Check if annotation exists
                const annotationKey = `${{pointData.x}}_${{pointData.y}}`;

                if (activeAnnotations[plotId].has(annotationKey)) {{
                    // Remove annotation
                    removeAnnotation(plotDiv, plotId, annotationKey, pointData);
                }} else {{
                    // Add annotation
                    addAnnotation(plotDiv, plotId, annotationKey, pointData);
                }}
            }});
        }});

        // Add click handler to BCG Portfolio Matrix
        // Wait for Plotly plot to be fully rendered before attaching event handler
        setTimeout(function() {{
            const portfolioDiv = document.getElementById('plot_portfolio');
            if (portfolioDiv && portfolioDiv.on) {{
                activeAnnotations['plot_portfolio'] = new Set();

                portfolioDiv.on('plotly_click', function(data) {{
                    const point = data.points[0];
                    if (!point) return;

                    // Get point metadata
                    const pointIndex = point.customdata ? point.customdata[0] : point.pointIndex;
                    const pointData = portfolioData[pointIndex];
                    if (!pointData) return;

                    // Check if annotation exists
                    const annotationKey = `${{pointData.x}}_${{pointData.y}}`;

                    if (activeAnnotations['plot_portfolio'].has(annotationKey)) {{
                        // Remove annotation
                        removeAnnotation(portfolioDiv, 'plot_portfolio', annotationKey, pointData);
                    }} else {{
                        // Add annotation
                        addAnnotation(portfolioDiv, 'plot_portfolio', annotationKey, pointData);
                    }}
                }});
                console.log('✓ BCG Portfolio Matrix click handler attached');
            }} else {{
                console.error('❌ BCG Portfolio Matrix not ready or not a Plotly plot');
            }}
        }}, 500);  // Wait 500ms for plots to render

        function addAnnotation(plotDiv, plotId, key, pointData) {{
            const currentLayout = plotDiv.layout;
            const annotations = currentLayout.annotations || [];

            // Create new annotation with leader line
            // Note: Plot-level config has editable:true, so annotations are draggable by default
            const newAnnotation = {{
                x: pointData.x,
                y: pointData.y,
                xref: 'x',
                yref: 'y',
                text: `<b>${{pointData.name}}</b><br>Category: ${{pointData.category}}<br>Source: ${{pointData.source}}`,
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: pointData.color,
                ax: 60,  // Arrow x offset (draggable via plot config)
                ay: -60, // Arrow y offset (draggable via plot config)
                bgcolor: 'rgba(255, 255, 255, 0.95)',
                bordercolor: pointData.color,
                borderwidth: 2,
                borderpad: 6,
                font: {{
                    size: 11,
                    color: '#000'
                }},
                align: 'left',
                captureevents: true,
                // Custom property to track which point this belongs to
                _key: key
            }};

            annotations.push(newAnnotation);
            activeAnnotations[plotId].add(key);

            Plotly.relayout(plotDiv, {{
                annotations: annotations
            }});
        }}

        function removeAnnotation(plotDiv, plotId, key, pointData) {{
            const currentLayout = plotDiv.layout;
            let annotations = currentLayout.annotations || [];

            // Filter out the annotation with matching key
            annotations = annotations.filter(ann => ann._key !== key);
            activeAnnotations[plotId].delete(key);

            Plotly.relayout(plotDiv, {{
                annotations: annotations
            }});
        }}

        console.log('✓ Click-to-pin annotations enabled on 2D plots and BCG Portfolio Matrix');
        console.log('💡 Click any point to pin/unpin a label. Drag pinned labels to reposition.');
        console.log('📌 Annotations are editable - click and drag the label box to move it.');
    </script>
</body>
</html>
"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"    ✓ Interactive dashboard saved to {output_file}")

    def _create_subcriteria_analysis(self):
        """Analyze subcriteria if available"""

        subcriteria_cols = [col for col in self.df.columns
                          if 'ease_adoption' in col or 'resources_required' in col or
                          'technical_complexity' in col or 'change_management' in col]

        if not subcriteria_cols or len(subcriteria_cols) < 1:
            print("    ⚠ No subcriteria data available")
            return

        # Use available subcriteria (up to 4)
        subcriteria_cols = subcriteria_cols[:4]

        n_plots = len(subcriteria_cols)
        n_rows = (n_plots + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()

        for idx, col in enumerate(subcriteria_cols):
            ax = axes[idx]

            # Create histogram
            ax.hist(self.df[col], bins=20, color='skyblue',
                   edgecolor='black', alpha=0.7)

            # Add statistics
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()

            ax.axvline(mean_val, color='red', linestyle='--',
                      label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='--',
                      label=f'Median: {median_val:.1f}')

            subcrit_name = col.replace('_', ' ').title()
            ax.set_title(f'{subcrit_name} Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(subcriteria_cols), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Implementation Difficulty Subcriteria Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        output_file = f'{self.output_path}subcriteria_analysis.png'
        plt.savefig(output_file, dpi=150)
        print(f"    ✓ Subcriteria analysis saved to {output_file}")
        plt.close()


def load_12d_results(json_path: str) -> pd.DataFrame:
    """Load 12D analysis results from JSON and convert to DataFrame"""

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract methods array
    methods = data['methods']

    # Convert to DataFrame
    df = pd.DataFrame(methods)

    print(f"✓ Loaded {len(df)} methods from {json_path}")

    return df


def visualize_scored_methods(json_path: str = None, df: pd.DataFrame = None):
    """
    Create visualizations from scored methods

    Args:
        json_path: Path to JSON file with scored methods
        df: DataFrame with scored methods (alternative to JSON)
    """

    if df is None and json_path:
        df = load_12d_results(json_path)
    elif df is None:
        print("Error: Provide either json_path or df")
        return

    print(f"\nCreating visualizations for {len(df)} methods...")

    dashboard = MethodVisualizationDashboard(df)
    dashboard.create_comprehensive_dashboard()

    print("\n" + "="*70)
    print("FILES GENERATED")
    print("="*70)
    print("  ✓ results/evaluation_dashboard.png (static)")
    print("  ✓ results/interactive_dashboard.html (interactive, 4 2D + 3 3D plots)")
    print("  ✓ results/subcriteria_analysis.png (if available)")
    print("\n💡 Open interactive_dashboard.html in a browser for best experience!")


def main():
    parser = argparse.ArgumentParser(
        description='Create comprehensive visualizations from 12D analysis results'
    )
    parser.add_argument(
        '--input',
        help='Input JSON file with 12D analysis results (default: auto-detect)'
    )
    args = parser.parse_args()

    # Auto-detect input file if not specified
    if args.input:
        input_path = Path(args.input)
    else:
        # Check for 12D results (deduplicated version first)
        dedup_path = Path("results/method_scores_12d_deduplicated.json")
        full_path = Path("results/method_scores_12d.json")

        if dedup_path.exists():
            input_path = dedup_path
            print("✓ Auto-detected deduplicated 12D results")
        elif full_path.exists():
            input_path = full_path
            print("✓ Auto-detected full 12D results")
        else:
            print("❌ Error: No 12D analysis results found")
            print("\nPlease run:")
            print("  python analyze_12d.py")
            return

    if not input_path.exists():
        print(f"❌ Error: {input_path} not found")
        return

    print("="*70)
    print("12D VISUALIZATION DASHBOARD")
    print("="*70)
    print(f"\nInput:  {input_path}")
    print(f"Output: results/")

    visualize_scored_methods(json_path=str(input_path))


if __name__ == "__main__":
    main()

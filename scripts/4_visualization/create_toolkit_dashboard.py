#!/usr/bin/env python3
"""
Scientific Toolkit Visualization - Data-Driven Selection Justification

Creates quantitative, visual explanations of WHY specific methods were selected.
Replaces verbose text with clean charts and data.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from cluster_utils import load_cluster_mappings, get_category_display_name


class ScientificToolkitVisualizer:
    """Generate scientific, data-driven visualizations for toolkit selections"""

    def __init__(self, toolkit_comparison: Dict, methods_df: pd.DataFrame):
        self.comparison = toolkit_comparison
        self.methods_df = methods_df
        self.contexts = list(toolkit_comparison['toolkits'].keys())
        # Load cluster mappings for human-readable category names
        self.cluster_mappings = load_cluster_mappings()

    def create_methods_table_separate(self, context: str) -> go.Figure:
        """Panel A: Methods table as standalone figure"""
        toolkit = self.comparison['toolkits'][context]
        methods = toolkit['methods']

        method_names = [m['name'] for m in methods]
        categories = [get_category_display_name(m.get('category', 'unknown'), self.cluster_mappings) for m in methods]
        fitness = [f"{m.get('fitness_score', m.get('context_fitness', 0)):.0f}/100" for m in methods]
        compat = [f"{m.get('avg_compatibility', 0):.3f}" for m in methods]
        impact = [f"{m.get('impact_potential', 0):.0f}/100" for m in methods]

        # Create table without height constraints to avoid scrollbars
        # Column widths: # (narrow), Method Name (wide), Category (medium), Fitness/Compat/Impact (narrow)
        fig = go.Figure(data=[go.Table(
            columnwidth=[30, 250, 180, 70, 85, 60],  # Relative widths for each column
            header=dict(
                values=['#', 'Method Name', 'Category', 'Fitness', 'Compat.', 'Impact'],
                fill_color='#667eea',
                font=dict(color='white', size=12),
                align=['center', 'left', 'left', 'center', 'center', 'center'],
                height=30
            ),
            cells=dict(
                values=[
                    list(range(1, len(methods)+1)),
                    method_names,
                    categories,
                    fitness,
                    compat,
                    impact
                ],
                fill_color='white',
                font=dict(size=11),
                align=['center', 'left', 'left', 'center', 'center', 'center'],
                height=28
            ),
            # CRITICAL: Set domain to use full figure height - no scrolling
            domain=dict(x=[0, 1], y=[0, 1])
        )])

        # Dynamic height - ensure all rows visible without scrollbars
        # Header (30px) + each row (28px) + margins
        table_height = 30 + (len(methods) * 28) + 150
        fig.update_layout(
            title="A. Selected Methods & Categories",
            height=table_height,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        return fig

    def create_implementation_roadmap_separate(self, context: str) -> go.Figure:
        """Panel B: Implementation roadmap as standalone figure"""
        toolkit = self.comparison['toolkits'][context]
        methods = toolkit['methods']

        difficulties = [m.get('implementation_difficulty', 50) for m in methods]
        time_values = [m.get('time_to_value', 50) for m in methods]
        impacts = [m.get('impact_potential', 50) for m in methods]

        hover_text = []
        text_colors = []
        for i, m in enumerate(methods):
            hover_text.append(
                f"<b>{i+1}. {m['name']}</b><br>" +
                f"Difficulty: {m.get('implementation_difficulty', 50):.0f}/100<br>" +
                f"Time to Value: {m.get('time_to_value', 50):.0f}/100<br>" +
                f"Impact: {m.get('impact_potential', 50):.0f}/100<br>"
            )
            impact = m.get('impact_potential', 50)
            if 25 <= impact <= 80:
                text_colors.append('black')
            else:
                text_colors.append('white')

        fig = go.Figure()

        for color in ['white', 'black']:
            indices = [i for i, c in enumerate(text_colors) if c == color]
            if not indices:
                continue

            fig.add_trace(go.Scatter(
                x=[difficulties[i] for i in indices],
                y=[time_values[i] for i in indices],
                mode='markers+text',
                text=[str(i+1) for i in indices],
                textposition='middle center',
                textfont=dict(color=color, size=10, family='Arial Black'),
                marker=dict(
                    size=[max(20, impacts[i]/3) for i in indices],
                    color=[impacts[i] for i in indices],
                    colorscale='RdYlGn',
                    showscale=(color == 'white'),
                    colorbar=dict(
                        title="Impact<br>Potential",
                        x=1.02,
                        len=0.85,
                        y=0.5,
                        yanchor='middle',
                        tickvals=[0, 25, 50, 75, 100],
                        ticktext=['0', '25', '50', '75', '100']
                    ) if color == 'white' else None,
                    line=dict(color='black', width=1)
                ),
                hovertext=[hover_text[i] for i in indices],
                hoverinfo='text',
                showlegend=False
            ))

        fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="gray", width=2, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=50, x1=100, y1=50, line=dict(color="gray", width=2, dash="dash"))

        fig.update_xaxes(title_text="← Easy | Implementation Difficulty | Hard →", range=[-5, 105], showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="← Slow | Time to Value | Fast →", range=[-5, 105], showgrid=True, gridcolor='lightgray')

        fig.update_layout(
            title="B. Implementation Roadmap",
            height=500,
            plot_bgcolor='white',
            margin=dict(l=80, r=120, t=60, b=80)
        )
        return fig

    def create_score_composition_table_separate(self, context: str) -> go.Figure:
        """Panel C: Score composition table as standalone figure"""
        toolkit = self.comparison['toolkits'][context]
        methods = toolkit['methods']

        score_components = []
        for i, m in enumerate(methods):
            fitness_contrib = m.get('fitness_score', m.get('context_fitness', 0)) * 0.4
            avg_compat_contrib = m.get('avg_compatibility', 0) * 100 * 0.25
            min_compat_contrib = m.get('min_compatibility', 0) * 100 * 0.20
            diversity_contrib = m.get('diversity_score', 0)
            synergy_contrib = m.get('synergy_score', 0) * 0.15
            total = fitness_contrib + avg_compat_contrib + min_compat_contrib + diversity_contrib + synergy_contrib

            score_components.append({
                'num': i+1,
                'fitness': f"{fitness_contrib:.1f}",
                'avg_compat': f"{avg_compat_contrib:.1f}",
                'min_compat': f"{min_compat_contrib:.1f}",
                'diversity': f"{diversity_contrib:.1f}",
                'synergy': f"{synergy_contrib:.1f}",
                'total': f"{total:.1f}"
            })

        # Create table without height constraints to avoid scrollbars
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['#', 'Fitness<br>(40%)', 'Avg Compat*<br>(25%)', 'Min Compat*<br>(20%)',
                       'Diversity<br>(10%)', 'Synergy<br>(15%)', 'Total'],
                fill_color='#764ba2',
                font=dict(color='white', size=11),
                align='center',
                height=30
            ),
            cells=dict(
                values=[
                    [s['num'] for s in score_components],
                    [s['fitness'] for s in score_components],
                    [s['avg_compat'] for s in score_components],
                    [s['min_compat'] for s in score_components],
                    [s['diversity'] for s in score_components],
                    [s['synergy'] for s in score_components],
                    [s['total'] for s in score_components]
                ],
                fill_color='white',
                font=dict(size=10),
                align='center',
                height=26
            ),
            # CRITICAL: Set domain to use full figure height - no scrolling
            domain=dict(x=[0, 1], y=[0, 1])
        )])

        # Dynamic height - ensure all rows visible without scrollbars
        # Header (30px) + each row (26px) + margins
        table_height = 30 + (len(methods) * 26) + 100
        fig.update_layout(
            title="C. Score Composition",
            height=table_height,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        return fig

    def create_methods_table(self, context: str) -> go.Figure:
        """
        Panel A: Methods list with categories (OLD VERSION - kept for compatibility)
        """
        toolkit = self.comparison['toolkits'][context]
        methods = toolkit['methods']

        # Panel A: Methods list with categories and key scores
        method_names = [m['name'] for m in methods]
        categories = [get_category_display_name(m.get('category', 'unknown'), self.cluster_mappings) for m in methods]
        fitness = [f"{m.get('fitness_score', m.get('context_fitness', 0)):.0f}/100" for m in methods]
        compat = [f"{m.get('avg_compatibility', 0):.3f}" for m in methods]
        impact = [f"{m.get('impact_potential', 0):.0f}/100" for m in methods]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['#', 'Method Name', 'Category', 'Fitness', 'Compatibility', 'Impact'],
                    fill_color='#667eea',
                    font=dict(color='white', size=12),
                    align='left'
                ),
                cells=dict(
                    values=[
                        list(range(1, len(methods)+1)),
                        method_names,
                        categories,
                        fitness,
                        compat,
                        impact
                    ],
                    fill_color='white',
                    font=dict(size=11),
                    align='left',
                    height=25
                )
            ),
            row=1, col=1
        )

        # Panel B: Implementation roadmap (clean 2D scatter with labels)
        difficulties = [m.get('implementation_difficulty', 50) for m in methods]
        time_values = [m.get('time_to_value', 50) for m in methods]
        impacts = [m.get('impact_potential', 50) for m in methods]

        # Use numbered markers with hover info
        hover_text = []
        text_colors = []
        for i, m in enumerate(methods):
            hover_text.append(
                f"<b>{i+1}. {m['name']}</b><br>" +
                f"Difficulty: {m.get('implementation_difficulty', 50):.0f}/100<br>" +
                f"Time to Value: {m.get('time_to_value', 50):.0f}/100<br>" +
                f"Impact: {m.get('impact_potential', 50):.0f}/100<br>"
            )

            # High contrast text: black only for light yellow middle, white for dark red/green ends
            # RdYlGn colorscale: red (0-40) → yellow (40-70) → green (70-100)
            impact = m.get('impact_potential', 50)
            if 40 <= impact <= 70:  # Yellow middle range (light) - use BLACK text
                text_colors.append('black')
            else:  # Red/Green ends (dark) - use WHITE text
                text_colors.append('black')

        # Create one trace per text color for proper rendering
        # Group by color to minimize traces
        for color in ['white', 'black']:
            indices = [i for i, c in enumerate(text_colors) if c == color]
            if not indices:
                continue

            fig.add_trace(
                go.Scatter(
                    x=[difficulties[i] for i in indices],
                    y=[time_values[i] for i in indices],
                    mode='markers+text',
                    text=[str(i+1) for i in indices],
                    textposition='middle center',
                    textfont=dict(color=color, size=10, family='Arial Black'),
                    marker=dict(
                        size=[max(20, impacts[i]/3) for i in indices],
                        color=[impacts[i] for i in indices],
                        colorscale='RdYlGn',
                        showscale=(color == 'white'),  # Show colorbar only once
                        colorbar=dict(
                            title="Impact<br>Potential",
                            x=1.02,
                            # Calculate exact position in paper coordinates
                            # Panel B starts at panel_a_height and spans panel_b_height
                            # Account for vertical_spacing (0.08 between panels)
                            len=panel_b_height - 0.05,  # Slightly less than full height for padding
                            y=panel_a_height + 0.04 + (panel_b_height / 2),  # Offset for spacing
                            yanchor='middle',
                            tickvals=[0, 25, 50, 75, 100],
                            ticktext=['0', '25', '50', '75', '100']
                        ) if color == 'black' else None,  # Show for white text group
                        line=dict(color='black', width=1)
                    ),
                    hovertext=[hover_text[i] for i in indices],
                    hoverinfo='text',
                    showlegend=False
                ),
                row=2, col=1
            )

        # Update roadmap axes with proper scale and gridlines
        fig.update_xaxes(
            title_text="← Easy | Implementation Difficulty | Hard →",
            row=2, col=1,
            range=[-5, 105],
            dtick=20,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        )
        fig.update_yaxes(
            title_text="← Slow | Time to Value | Fast →",
            row=2, col=1,
            range=[-5, 105],
            dtick=20,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        )

        # Add quadrant reference lines
        fig.add_shape(
            type="line",
            x0=0, x1=100, y0=50, y1=50,
            line=dict(color="gray", width=2, dash="dash"),
            row=2, col=1
        )
        fig.add_shape(
            type="line",
            x0=50, x1=50, y0=0, y1=100,
            line=dict(color="gray", width=2, dash="dash"),
            row=2, col=1
        )

        # Panel C: Scoring formula breakdown
        score_components = []
        for i, m in enumerate(methods):
            fitness_contrib = m.get('fitness_score', m.get('context_fitness', 0)) * 0.4
            avg_compat_contrib = m.get('avg_compatibility', 0) * 100 * 0.25
            min_compat_contrib = m.get('min_compatibility', 0) * 100 * 0.20

            # Use ACTUAL diversity and synergy scores from toolkit builder
            diversity_contrib = m.get('diversity_score', 0)  # Already 0-10 scale
            synergy_contrib = m.get('synergy_score', 0) * 0.15  # Scale from 0-100 to contribution

            total = fitness_contrib + avg_compat_contrib + min_compat_contrib + diversity_contrib + synergy_contrib

            score_components.append({
                'num': i+1,
                'fitness': f"{fitness_contrib:.1f}",
                'avg_compat': f"{avg_compat_contrib:.1f}",
                'min_compat': f"{min_compat_contrib:.1f}",
                'diversity': f"{diversity_contrib:.1f}",
                'synergy': f"{synergy_contrib:.1f}",
                'total': f"{total:.1f}"
            })

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['#', 'Fitness<br>(40%)', 'Avg Compat<br>(25%)', 'Min Compat<br>(20%)',
                           'Diversity<br>(10%)', 'Synergy<br>(15%)', 'Total'],
                    fill_color='#764ba2',
                    font=dict(color='white', size=11),
                    align='center'
                ),
                cells=dict(
                    values=[
                        [s['num'] for s in score_components],
                        [s['fitness'] for s in score_components],
                        [s['avg_compat'] for s in score_components],
                        [s['min_compat'] for s in score_components],
                        [s['diversity'] for s in score_components],
                        [s['synergy'] for s in score_components],
                        [s['total'] for s in score_components]
                    ],
                    fill_color='white',
                    font=dict(size=10),
                    align='center',
                    height=22
                )
            ),
            row=3, col=1
        )

        # Calculate dynamic height based on number of methods to avoid scrollbars
        # Each method needs ~25-30px in tables, plus panel B needs space for plot
        min_height = 1200
        dynamic_height = max(min_height, 800 + (num_methods * 35))

        fig.update_layout(
            title_text=f"{toolkit['context_name']} - Toolkit",
            height=dynamic_height,
            showlegend=False,
            # Add explicit margins to ensure colorbar doesn't overlap
            margin=dict(l=50, r=120, t=80, b=50)
        )

        return fig

    def create_selection_formula_breakdown(self, context: str, method_index: int = 0) -> go.Figure:
        """
        Panel 2: Waterfall chart showing exact scoring formula
        """

        toolkit = self.comparison['toolkits'][context]
        methods = toolkit['methods']

        if method_index >= len(methods):
            method_index = 0

        method = methods[method_index]

        # Calculate score components using ACTUAL data
        fitness = method.get('fitness_score', method.get('context_fitness', 50)) * 0.4
        avg_compat = method.get('avg_compatibility', 0.8) * 100 * 0.25
        min_compat = method.get('min_compatibility', 0.7) * 100 * 0.20
        diversity = method.get('diversity_score', 0)  # Actual diversity score
        synergy = method.get('synergy_score', 0) * 0.15  # Actual synergy score

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", "total"],
            x=["Base Fitness\n(40%)", "Avg Compatibility\n(25%)", "Min Compatibility\n(20%)",
               "Category Diversity\n(10%)", "Synergy Bonus\n(15%)", "Total Score"],
            textposition="outside",
            y=[fitness, avg_compat, min_compat, diversity, synergy, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            increasing={"marker": {"color": "#2ecc71"}},
            totals={"marker": {"color": "#3498db"}}
        ))

        fig.update_layout(
            title=f"Selection Score Breakdown: {method['name']}",
            showlegend=False,
            height=500,
            yaxis_title="Score Contribution"
        )

        return fig

    def create_context_comparison_heatmap(self) -> go.Figure:
        """
        Panel 3: Heatmap showing which methods appear in which contexts
        Methods grouped by category horizontally with category headers
        FIXED: No gaps, proper alignment, no empty columns
        """

        # Collect all methods with their categories
        method_category_map = {}
        for toolkit in self.comparison['toolkits'].values():
            for method in toolkit['methods']:
                method_category_map[method['name']] = method.get('category', 'unknown')

        # Define category order (47 semantic clusters: P0-P20, S0-S25, U)
        # Primary clusters first, then secondary, then uncategorized
        category_order = (
            [f'P{i}' for i in range(21)] +  # P0-P20
            [f'S{i}' for i in range(26)] +  # S0-S25
            ['U', 'unknown']
        )

        # Group methods by category
        methods_by_category = {cat: [] for cat in category_order}
        for method_name, category in method_category_map.items():
            if category not in methods_by_category:
                methods_by_category['unknown'].append(method_name)
            else:
                methods_by_category[category].append(method_name)

        # Sort methods within each category alphabetically
        for cat in methods_by_category:
            methods_by_category[cat].sort()

        # Build ordered method list (grouped by category WITHOUT gaps)
        ordered_methods = []
        category_labels = []  # Each method gets its category label
        category_boundaries = []
        current_pos = 0

        for cat in category_order:
            if methods_by_category[cat]:
                cat_methods = methods_by_category[cat]
                category_boundaries.append({
                    'category': cat,
                    'start': current_pos,
                    'end': current_pos + len(cat_methods) - 1,
                    'count': len(cat_methods)
                })
                ordered_methods.extend(cat_methods)

                # Each method gets the same category label
                cat_name = cat.replace('_', ' ').title()
                category_labels.extend([cat_name] * len(cat_methods))

                current_pos += len(cat_methods)
                # NO GAP ADDED

        contexts = [self.comparison['toolkits'][c]['context_name'] for c in self.contexts]

        # Create binary matrix for contexts (no empty columns)
        matrix = []
        for context_key in self.contexts:
            toolkit = self.comparison['toolkits'][context_key]
            row = [1 if m in toolkit['method_names'] else 0 for m in ordered_methods]
            matrix.append(row)

        # Create figure with two rows: category header + main heatmap
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.05, 0.95],
            vertical_spacing=0.02,
            specs=[[{'type': 'heatmap'}], [{'type': 'heatmap'}]]
        )

        # Row 1: Category header row - show category for EACH method
        fig.add_trace(
            go.Heatmap(
                z=[list(range(len(ordered_methods)))],  # Color by position for visual grouping
                x=ordered_methods,
                y=['Category'],
                text=[category_labels],
                texttemplate='%{text}',
                textfont=dict(size=9, color='black'),
                showscale=False,
                colorscale=[[0, '#f0f0f0'], [1, '#f0f0f0']],  # Uniform gray background
                hoverinfo='skip',
                xaxis='x1',
                yaxis='y1'
            ),
            row=1, col=1
        )

        # Row 2: Main selection heatmap
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=ordered_methods,
                y=contexts,
                colorscale=[[0, 'white'], [1, '#2ecc71']],
                showscale=False,
                text=matrix,
                texttemplate='%{text}',
                textfont={'size': 9},
                xaxis='x2',
                yaxis='y2',
                hovertemplate='<b>%{x}</b><br>Context: %{y}<br>Selected: %{z}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add vertical lines to separate categories
        shapes = []
        for i, boundary in enumerate(category_boundaries[:-1]):
            x_position = boundary['end'] + 0.5
            
            # Line in category header
            shapes.append(dict(
                type='line',
                x0=x_position, x1=x_position,
                y0=0, y1=1,
                xref='x1', yref='paper',
                line=dict(color='#333', width=2)
            ))
            # Line in main heatmap
            shapes.append(dict(
                type='line',
                x0=x_position, x1=x_position,
                y0=0, y1=0.95,
                xref='x2', yref='paper',
                line=dict(color='#333', width=2)
            ))

        # Update axes
        fig.update_xaxes(
            tickangle=45,
            side='bottom',
            tickfont=dict(size=8),
            showticklabels=True,
            row=2, col=1
        )
        fig.update_xaxes(
            showticklabels=False,
            row=1, col=1
        )
        fig.update_yaxes(
            tickfont=dict(size=9),
            row=2, col=1
        )
        fig.update_yaxes(
            tickfont=dict(size=9),
            showticklabels=False,
            row=1, col=1
        )

        # Update layout with tighter margins to reduce gray space
        fig.update_layout(
            title='Method Selection Across Contexts - Grouped by Category<br><sub>(1=Selected, 0=Not Selected)</sub>',
            height=500,
            showlegend=False,
            shapes=shapes,
            margin=dict(t=80, b=150, l=150, r=20)  # Reduced right margin from 50 to 20
        )

        return fig

    def generate_clean_summary(self, context: str) -> str:
        """
        Panel 4: Clean, data-focused summary report
        """

        toolkit = self.comparison['toolkits'][context]
        methods = toolkit['methods']
        stats = toolkit['statistics']

        # Group by implementation wave
        waves = {
            'Immediate (0-3 months)': [],
            'Short-term (3-6 months)': [],
            'Long-term (6+ months)': []
        }

        for method in methods:
            diff = method.get('implementation_difficulty', 50)
            if diff < 40:
                wave = 'Immediate (0-3 months)'
            elif diff < 65:
                wave = 'Short-term (3-6 months)'
            else:
                wave = 'Long-term (6+ months)'

            waves[wave].append(method)

        report = f"""# TOOLKIT: {toolkit['context_name']}

## SELECTED METHODS ({toolkit['size']})

"""

        for wave_name, wave_methods in waves.items():
            if wave_methods:
                report += f"### {wave_name}\n\n"
                for m in sorted(wave_methods, key=lambda x: -x.get('impact_potential', 0)):
                    impact = m.get('impact_potential', 0)
                    compat = m.get('avg_compatibility', 0)
                    cat = m.get('category', 'unknown')
                    report += f"- **{m['name']}** | Category: {cat} | Impact: {impact:.0f}/100 | Compatibility: {compat:.2f}\n"
                report += "\n"

        # Key metrics
        report += f"""## KEY METRICS

- **Average Compatibility**: {stats['avg_compatibility']:.3f}
- **Category Coverage**: {stats['categories_covered']}/13
- **Implementation Risk**: {100 - stats.get('avg_difficulty', 50):.1f}/100
- **Average Impact**: {stats.get('avg_impact', 0):.1f}/100
- **Average Time to Value**: {stats.get('avg_time_to_value', 0):.1f}/100

## CATEGORY DISTRIBUTION

"""

        cat_dist = stats.get('category_distribution', {})
        for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
            report += f"- {cat}: {count} methods\n"

        return report

    def create_comprehensive_dashboard(self, context: str) -> go.Figure:
        """
        Create comprehensive 6-panel scientific dashboard
        """

        toolkit = self.comparison['toolkits'][context]
        methods = toolkit['methods']

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Method Selection Scores',
                'Compatibility Network',
                'Implementation Roadmap',
                'Category Balance',
                'Risk vs Impact Matrix',
                'Selection Drivers'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'pie'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )

        method_names = [m['name'][:20] for m in methods]

        # Panel 1: Selection scores
        fitness = [m.get('fitness_score', m.get('context_fitness', 0)) for m in methods]
        fig.add_trace(
            go.Bar(name='Fitness', x=method_names, y=fitness, marker_color='#667eea'),
            row=1, col=1
        )

        # Panel 2: Compatibility network (simplified)
        x_pos = list(range(len(methods)))
        y_pos = [m.get('avg_compatibility', 0) * 100 for m in methods]
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='markers+lines',
                marker=dict(size=10, color='#764ba2'),
                showlegend=False
            ),
            row=1, col=2
        )

        # Panel 3: Implementation roadmap
        difficulty = [m.get('implementation_difficulty', 50) for m in methods]
        time_val = [m.get('time_to_value', 50) for m in methods]
        impact = [m.get('impact_potential', 50) for m in methods]

        fig.add_trace(
            go.Scatter(
                x=difficulty,
                y=time_val,
                mode='markers',
                marker=dict(size=[i/3 for i in impact], color=impact, colorscale='Viridis'),
                text=method_names,
                showlegend=False
            ),
            row=2, col=1
        )

        # Panel 4: Category balance
        categories = [get_category_display_name(m.get('category', 'unknown'), self.cluster_mappings) for m in methods]
        cat_counts = pd.Series(categories).value_counts()

        fig.add_trace(
            go.Pie(labels=cat_counts.index, values=cat_counts.values, showlegend=False),
            row=2, col=2
        )

        # Panel 5: Risk vs Impact
        risk = [100 - m.get('implementation_difficulty', 50) for m in methods]
        fig.add_trace(
            go.Scatter(
                x=risk,
                y=impact,
                mode='markers+text',
                text=[str(i+1) for i in range(len(methods))],
                marker=dict(size=12, color='#e74c3c'),
                showlegend=False
            ),
            row=3, col=1
        )

        # Panel 6: Selection drivers (primary reason for selection)
        drivers = {'Fitness': 0, 'Compatibility': 0, 'Impact': 0, 'Speed': 0}
        for m in methods:
            scores = {
                'Fitness': m.get('fitness_score', m.get('context_fitness', 0)),
                'Compatibility': m.get('avg_compatibility', 0) * 100,
                'Impact': m.get('impact_potential', 0),
                'Speed': m.get('time_to_value', 0)
            }
            primary = max(scores, key=scores.get)
            drivers[primary] += 1

        fig.add_trace(
            go.Bar(
                x=list(drivers.keys()),
                y=list(drivers.values()),
                marker_color='#3498db',
                showlegend=False
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f"{toolkit['context_name']} - Toolkit overview",
            height=1200,
            showlegend=False
        )

        # Update axes
        fig.update_xaxes(title_text="Methods", tickangle=45, row=1, col=1)
        fig.update_xaxes(title_text="Difficulty →", row=2, col=1)
        fig.update_xaxes(title_text="Ease (100-Difficulty) →", row=3, col=1)
        fig.update_xaxes(title_text="Selection Driver", row=3, col=2)

        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Compatibility %", row=1, col=2)
        fig.update_yaxes(title_text="Time to Value →", row=2, col=1)
        fig.update_yaxes(title_text="Impact →", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=2)

        return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate scientific visualizations for toolkit selections'
    )
    parser.add_argument(
        '--comparison',
        default='results/toolkit_comparison.json',
        help='Toolkit comparison JSON file'
    )
    parser.add_argument(
        '--methods',
        default='results/method_scores_12d_deduplicated.json',
        help='Methods with 12D scores'
    )
    parser.add_argument(
        '--output',
        default='results/',
        help='Output directory'
    )

    args = parser.parse_args()

    # Load data
    with open(args.comparison, 'r') as f:
        comparison = json.load(f)

    with open(args.methods, 'r') as f:
        methods_data = json.load(f)
        methods_df = pd.DataFrame(methods_data['methods'])

    viz = ScientificToolkitVisualizer(comparison, methods_df)
    output_dir = Path(args.output)

    print("=" * 70)
    print("SCIENTIFIC TOOLKIT VISUALIZATION")
    print("=" * 70)

    # Generate clean visualizations for each context
    for context in viz.contexts:
        context_name = viz.comparison['toolkits'][context]['context_name']
        print(f"\nGenerating for: {context_name}")

        # Create 3 separate figures for better layout control
        toolkit = viz.comparison['toolkits'][context]
        methods = toolkit['methods']

        # Figure A: Methods table
        fig_a = viz.create_methods_table_separate(context)
        # Figure B: Implementation roadmap
        fig_b = viz.create_implementation_roadmap_separate(context)
        # Figure C: Score composition table
        fig_c = viz.create_score_composition_table_separate(context)

        # Combine into single HTML with divs
        # Get HTML for each plot (include plotly.js only once)
        html_a = fig_a.to_html(full_html=False, include_plotlyjs=False, div_id="methods-table")
        html_b = fig_b.to_html(full_html=False, include_plotlyjs=False, div_id="roadmap-plot")
        html_c = fig_c.to_html(full_html=False, include_plotlyjs=False, div_id="scores-table")

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{context_name} - Toolkit</title>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 920;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-container {{
            margin: 20px 0;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            overflow: visible;
            height: auto;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{context_name} - Toolkit</h1>

        <div class="plot-container">
            {html_a}
        </div>

        <div class="plot-container">
            {html_b}
        </div>

        <div class="plot-container">
            {html_c}
        </div>

        <div class="plot-container" style="background-color: #f9f9f9;">
            <h2 style="color: #667eea; margin-top: 0;">Selection Criteria Explanation</h2>

            <div style="margin: 15px 0;">
                <h4 style="color: #667eea; margin-bottom: 8px;">Fitness (40% weight)</h4>
                <ul style="line-height: 1.8;">
                    <li><strong>What it measures:</strong> How well the method fits your specific organizational context</li>
                    <li><strong>Calculation:</strong> Each method has pre-scored dimensions (0-100 scale). The fitness calculation uses a weighted sum of dimensions available for your context:
                        <ul style="margin-top: 8px; margin-bottom: 8px;">
                            <li><strong>Startup - MVP Phase (5 dimensions):</strong> time_to_value (0.35), ease_adoption (0.25), resources_required (-0.20), impact_potential (0.15), applicability (0.05)</li>
                            <li><strong>Startup - Scaling Phase (6 dimensions):</strong> scope (0.20), impact_potential (0.25), process_focus (0.15), time_to_value (0.20), technical_complexity (-0.10), change_management_difficulty (-0.10)</li>
                            <li><strong>Enterprise - Digital Transformation (4 dimensions):</strong> scope (0.30), temporality (0.25), impact_potential (0.25), change_management_difficulty (-0.20)</li>
                            <li><strong>Regulated Industry - Compliance Focus (4 dimensions):</strong> process_focus (0.30), technical_complexity (0.10), applicability (0.20), ease_adoption (0.15)
                            <br><em>Note: risk_decision_making (0.25) defined in code but dimension not available in data, so ignored</em></li>
                            <li><strong>Hardware Product Development (2 dimensions):</strong> time_to_value (-0.10), impact_potential (0.20)
                            <br><em>Note: planning_adaptation (0.25), risk_decision_making (0.25), design_development (0.20) defined in code but dimensions not available in data, so ignored</em></li>
                        </ul>
                        Negative weights mean "lower is better" (e.g., -0.20 for resources_required in MVP context means methods requiring fewer resources score higher).
                        The weighted sum is normalized to 0-100 scale across all methods.
                    </li>
                    <li><strong>Scale:</strong> Final fitness score 0-100, multiplied by 0.4 for contribution (0-40 points)</li>
                    <li><strong>Example:</strong> In Startup MVP context, a method with time_to_value=90, ease_adoption=85, resources_required=20 (low resources=high score), impact_potential=75, applicability=80 would score high on fitness because it aligns with MVP priorities (fast results, easy to adopt, minimal resources needed).</li>
                </ul>
            </div>

            <div style="margin: 15px 0;">
                <h4 style="color: #667eea; margin-bottom: 8px;">Avg Compat* (25% weight)</h4>
                <ul style="line-height: 1.8;">
                    <li><strong>What it measures:</strong> Average compatibility score with all other methods already in the toolkit</li>
                    <li><strong>Why average:</strong> Shows overall harmony - method should work well with the entire toolkit, not just one or two methods</li>
                    <li><strong>Calculation:</strong> Mean of pairwise compatibility scores (0-1 scale) × 100 × 0.25</li>
                    <li><strong>Scale:</strong> 0-25 points (higher = better overall fit with toolkit)</li>
                </ul>
            </div>

            <div style="margin: 15px 0;">
                <h4 style="color: #667eea; margin-bottom: 8px;">Min Compat* (20% weight)</h4>
                <ul style="line-height: 1.8;">
                    <li><strong>What it measures:</strong> Minimum compatibility score with any single method in toolkit</li>
                    <li><strong>Why minimum:</strong> Ensures no weak links - prevents adding methods that conflict with existing ones</li>
                    <li><strong>Calculation:</strong> Lowest pairwise compatibility score × 100 × 0.20</li>
                    <li><strong>Scale:</strong> 0-20 points (penalizes methods with any problematic relationships)</li>
                </ul>
            </div>

            <div style="margin: 15px 0;">
                <h4 style="color: #667eea; margin-bottom: 8px;">Diversity (10% weight, up to 10 points)</h4>
                <ul style="line-height: 1.8;">
                    <li><strong>What it measures:</strong> Whether this method adds a new category to the toolkit</li>
                    <li><strong>Calculation:</strong> 10 points if new category, 0 points if category already present</li>
                    <li><strong>Why it matters:</strong> Encourages balanced coverage across different method types (iterative, planning, collaboration, etc.)</li>
                    <li><strong>Example:</strong> If toolkit has 3 iterative methods but no planning methods, a planning method gets +10</li>
                </ul>
            </div>

            <div style="margin: 15px 0;">
                <h4 style="color: #667eea; margin-bottom: 8px;">Synergy (15% weight)</h4>
                <ul style="line-height: 1.8;">
                    <li><strong>What it measures:</strong> Actual pairwise synergies and conflicts with toolkit methods (from LLM overlap analysis)</li>
                    <li><strong>Calculation breakdown:</strong>
                        <ul style="margin-top: 8px;">
                            <li>+0.3 per synergistic pair (score ≥0.95, not conflicting)</li>
                            <li>+0.15 per complementary pair (same problem, different approach, no conflict)</li>
                            <li>-0.2 per problematic overlap (conflicts detected)</li>
                        </ul>
                    </li>
                    <li><strong>Scale:</strong> Normalized to -0.5 to +1.0, then ×100×0.15 = -7.5 to +15 points</li>
                    <li><strong>Example:</strong> Method with 2 synergies and 1 conflict: (0.3+0.3-0.2)/3 × 100 × 0.15 ≈ +2 points</li>
                </ul>
            </div>

            <p style="margin-top: 25px; padding: 15px; background-color: #e8f0fe; border-radius: 5px; font-size: 13px;">
                <strong>Note:</strong> * "Avg Compat" and "Min Compat" are calculated with other methods already in toolkit at time of selection.
                First method has Compat=100 and Synergy=0 (nothing to compare with yet).
            </p>
        </div>
    </div>
</body>
</html>
"""

        output_file = output_dir / f'toolkit_{context}_analysis.html'
        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"  ✓ Analysis (methods list + roadmap + scoring)")

    # Context comparison
    fig_comp = viz.create_context_comparison_heatmap()
    fig_comp.write_html(output_dir / 'toolkit_context_comparison.html')
    print(f"\n✓ Context comparison heatmap")

    print("\n" + "=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    print(f"\nPer context (5 files):")
    for context in viz.contexts:
        print(f"  - toolkit_{context}_analysis.html")
    print(f"\nComparison:")
    print(f"  - toolkit_context_comparison.html")
    print("\n💡 Open any analysis file in browser to see method list, roadmap & scoring")


if __name__ == "__main__":
    main()
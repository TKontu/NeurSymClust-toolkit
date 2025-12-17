"""
Visualization module for methods analysis results.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

logger = logging.getLogger(__name__)


class AnalysisVisualizer:
    """Creates interactive visualizations of analysis results."""

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_all_visualizations(
        self,
        methods: List,
        duplicate_report: Dict,
        compatibility_report: Dict,
        abstraction_report: Dict,
        category_report: Dict = None,
        toolkit_report: Dict = None
    ):
        """Generate all visualization plots."""
        logger.info("Creating visualizations...")

        plots = []

        # 1. Category distribution pie chart
        if category_report:
            fig = self._plot_category_distribution(category_report)
            path = self.output_dir / "viz_category_distribution.html"
            fig.write_html(str(path))
            plots.append(path)
            logger.info(f"Created category distribution plot: {path}")

        # 2. Abstraction levels bar chart
        fig = self._plot_abstraction_levels(abstraction_report)
        path = self.output_dir / "viz_abstraction_levels.html"
        fig.write_html(str(path))
        plots.append(path)
        logger.info(f"Created abstraction levels plot: {path}")

        # 3. Toolkit comparison (if multiple toolkits)
        if toolkit_report and len(toolkit_report['toolkits']) > 0:
            fig = self._plot_toolkit_comparison(toolkit_report)
            path = self.output_dir / "viz_toolkit_comparison.html"
            fig.write_html(str(path))
            plots.append(path)
            logger.info(f"Created toolkit comparison plot: {path}")

        # 4. Individual toolkit visualizations
        if toolkit_report and category_report:
            for toolkit in toolkit_report['toolkits'][:5]:  # Top 5 toolkits
                fig = self._plot_single_toolkit(toolkit, category_report)
                path = self.output_dir / f"viz_toolkit_{toolkit['toolkit_id']}.html"
                fig.write_html(str(path))
                plots.append(path)
                logger.info(f"Created toolkit #{toolkit['toolkit_id']} visualization: {path}")

        # 5. Compatibility network graph
        if compatibility_report:
            fig = self._plot_compatibility_network(
                methods, compatibility_report, category_report
            )
            path = self.output_dir / "viz_compatibility_network.html"
            fig.write_html(str(path))
            plots.append(path)
            logger.info(f"Created compatibility network plot: {path}")

        # 6. Duplicate clusters network
        if duplicate_report['duplicate_groups']:
            fig = self._plot_duplicate_clusters(duplicate_report)
            path = self.output_dir / "viz_duplicate_clusters.html"
            fig.write_html(str(path))
            plots.append(path)
            logger.info(f"Created duplicate clusters plot: {path}")

        # 7. Source distribution
        fig = self._plot_source_distribution(methods)
        path = self.output_dir / "viz_source_distribution.html"
        fig.write_html(str(path))
        plots.append(path)
        logger.info(f"Created source distribution plot: {path}")

        # 8. Dashboard summary
        fig = self._create_dashboard(
            methods, duplicate_report, compatibility_report,
            abstraction_report, category_report, toolkit_report
        )
        path = self.output_dir / "viz_dashboard.html"
        fig.write_html(str(path))
        plots.append(path)
        logger.info(f"Created dashboard summary: {path}")

        return plots

    def _plot_category_distribution(self, category_report: Dict) -> go.Figure:
        """Pie chart of methods by category."""
        distribution = category_report['summary']['distribution']

        # Map category IDs to names
        category_names = []
        for cat_id, methods_list in category_report['by_category'].items():
            if methods_list:
                category_names.append(methods_list[0]['category_name'])
            else:
                category_names.append(cat_id)

        fig = go.Figure(data=[go.Pie(
            labels=category_names,
            values=list(distribution.values()),
            hole=0.3,
            textinfo='label+percent+value',
            textposition='auto'
        )])

        fig.update_layout(
            title="Methods Distribution by Category",
            height=600,
            showlegend=True
        )

        return fig

    def _plot_abstraction_levels(self, abstraction_report: Dict) -> go.Figure:
        """Bar chart of abstraction levels."""
        summary = abstraction_report['summary']

        categories = ['High (Principles)', 'Medium (Frameworks)', 'Low (Techniques)']
        values = [
            summary['high_level_count'],
            summary['medium_level_count'],
            summary['low_level_count']
        ]

        colors = ['#3498db', '#f39c12', '#e74c3c']

        fig = go.Figure(data=[go.Bar(
            x=categories,
            y=values,
            text=values,
            textposition='auto',
            marker_color=colors
        )])

        fig.update_layout(
            title="Methods by Abstraction Level",
            xaxis_title="Abstraction Level",
            yaxis_title="Number of Methods",
            height=500
        )

        return fig

    def _plot_toolkit_comparison(self, toolkit_report: Dict) -> go.Figure:
        """Compare multiple toolkits side by side."""
        toolkits = toolkit_report['toolkits']

        if len(toolkits) == 0:
            # Empty plot
            fig = go.Figure()
            fig.add_annotation(text="No toolkits generated", showarrow=False)
            return fig

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Toolkit Sizes',
                'Abstraction Mix',
                'Category Diversity',
                'Toolkit Scores'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )

        toolkit_ids = [f"Toolkit {t['toolkit_id']}" for t in toolkits]

        # 1. Sizes
        sizes = [t['size'] for t in toolkits]
        fig.add_trace(
            go.Bar(x=toolkit_ids, y=sizes, name="Size", marker_color='#3498db'),
            row=1, col=1
        )

        # 2. Abstraction mix (stacked)
        high = [t['abstraction_mix']['high'] for t in toolkits]
        medium = [t['abstraction_mix']['medium'] for t in toolkits]
        low = [t['abstraction_mix']['low'] for t in toolkits]

        fig.add_trace(
            go.Bar(x=toolkit_ids, y=high, name="High", marker_color='#3498db'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=toolkit_ids, y=medium, name="Medium", marker_color='#f39c12'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=toolkit_ids, y=low, name="Low", marker_color='#e74c3c'),
            row=1, col=2
        )

        # 3. Category diversity
        diversity = [len(t.get('category_distribution', {})) for t in toolkits]
        fig.add_trace(
            go.Bar(x=toolkit_ids, y=diversity, name="Categories", marker_color='#27ae60'),
            row=2, col=1
        )

        # 4. Scores
        scores = [t['score'] for t in toolkits]
        fig.add_trace(
            go.Bar(x=toolkit_ids, y=scores, name="Score", marker_color='#9b59b6'),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Toolkit Comparison Dashboard",
            showlegend=True,
            barmode='stack'
        )

        return fig

    def _plot_single_toolkit(self, toolkit: Dict, category_report: Dict) -> go.Figure:
        """Detailed visualization of a single toolkit."""
        methods = toolkit['methods']

        # Create sunburst chart: categories -> methods
        labels = []
        parents = []
        values = []
        colors = []

        # Root
        labels.append(f"Toolkit {toolkit['toolkit_id']}")
        parents.append("")
        values.append(0)
        colors.append("#ecf0f1")

        # Group by category
        category_groups = {}
        for method in methods:
            cat = method['category']
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(method)

        # Add categories
        color_palette = px.colors.qualitative.Set3
        for i, (cat, cat_methods) in enumerate(category_groups.items()):
            labels.append(cat)
            parents.append(f"Toolkit {toolkit['toolkit_id']}")
            values.append(len(cat_methods))
            colors.append(color_palette[i % len(color_palette)])

            # Add methods
            for method in cat_methods:
                labels.append(method['name'])
                parents.append(cat)
                values.append(1)
                colors.append(color_palette[i % len(color_palette)])

        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            branchvalues="total"
        ))

        fig.update_layout(
            title=f"Toolkit #{toolkit['toolkit_id']} - {toolkit['size']} Methods (Score: {toolkit['score']:.2f})",
            height=700
        )

        return fig

    def _plot_compatibility_network(
        self,
        methods: List,
        compatibility_report: Dict,
        category_report: Dict = None
    ) -> go.Figure:
        """Network graph showing method compatibility."""
        # Build graph
        G = nx.Graph()

        # Add nodes
        method_dict = {m.index: m for m in methods}
        for method in methods:
            G.add_node(method.index, name=method.name)

        # Add edges (high compatibility only)
        high_compat = [r for r in compatibility_report['all_results']
                      if r['compatibility_score'] >= 0.7]

        for result in high_compat[:200]:  # Limit to top 200 edges for clarity
            G.add_edge(
                result['method1_index'],
                result['method2_index'],
                weight=result['compatibility_score']
            )

        if G.number_of_nodes() == 0:
            fig = go.Figure()
            fig.add_annotation(text="No compatibility data available", showarrow=False)
            return fig

        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]].get('weight', 0.5)

            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight * 2, color='rgba(125, 125, 125, 0.3)'),
                hoverinfo='none',
                showlegend=False
            ))

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []

        # Build index to category mapping from category_report
        index_to_category = {}
        if category_report and 'by_category' in category_report:
            for cat_id, cat_methods in category_report['by_category'].items():
                for method_info in cat_methods:
                    index_to_category[method_info['index']] = cat_id

        # Color by category if available
        category_colors = {}
        if category_report:
            color_palette = px.colors.qualitative.Set3
            for i, cat_id in enumerate(category_report.get('by_category', {}).keys()):
                category_colors[cat_id] = color_palette[i % len(color_palette)]

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            method = method_dict[node]
            node_text.append(f"{method.name}<br>Connections: {G.degree(node)}")

            # Color by category
            if node in index_to_category:
                cat_id = index_to_category[node]
                node_color.append(category_colors.get(cat_id, '#95a5a6'))
            else:
                node_color.append('#95a5a6')

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )

        # Combine traces
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title="Method Compatibility Network (>0.7 compatibility)",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            plot_bgcolor='white'
        )

        return fig

    def _plot_duplicate_clusters(self, duplicate_report: Dict) -> go.Figure:
        """Network visualization of duplicate clusters."""
        groups = duplicate_report['duplicate_groups']

        if not groups:
            fig = go.Figure()
            fig.add_annotation(text="No duplicate groups found", showarrow=False)
            return fig

        # Build graph
        G = nx.Graph()

        for group in groups:
            methods = list(group['methods'])
            # Fully connect methods in same group
            for i, m1 in enumerate(methods):
                G.add_node(m1)
                for m2 in methods[i+1:]:
                    G.add_edge(m1, m2)

        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='#e74c3c'),
            hoverinfo='none'
        )

        # Node traces (color by group)
        node_traces = []
        colors = px.colors.qualitative.Set3

        for i, group in enumerate(groups):
            node_x = []
            node_y = []
            node_text = []

            for method in group['methods']:
                x, y = pos[method]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{method}<br>Group {group['group_id']}")

            node_traces.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[m.split()[0] for m in group['methods']],  # First word only
                textposition='top center',
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(size=15, color=colors[i % len(colors)]),
                name=f"Group {group['group_id']}"
            ))

        fig = go.Figure(data=[edge_trace] + node_traces)

        fig.update_layout(
            title=f"Duplicate Clusters ({len(groups)} groups)",
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            plot_bgcolor='white'
        )

        return fig

    def _plot_source_distribution(self, methods: List) -> go.Figure:
        """Bar chart of methods by source."""
        source_counts = {}
        for method in methods:
            source_counts[method.source] = source_counts.get(method.source, 0) + 1

        # Sort by count
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        sources = [s[0] for s in sorted_sources]
        counts = [s[1] for s in sorted_sources]

        fig = go.Figure(data=[go.Bar(
            x=sources,
            y=counts,
            text=counts,
            textposition='auto',
            marker_color='#3498db'
        )])

        fig.update_layout(
            title="Methods by Source",
            xaxis_title="Source",
            yaxis_title="Number of Methods",
            height=500,
            xaxis_tickangle=-45
        )

        return fig

    def _create_dashboard(
        self,
        methods: List,
        duplicate_report: Dict,
        compatibility_report: Dict,
        abstraction_report: Dict,
        category_report: Dict = None,
        toolkit_report: Dict = None
    ) -> go.Figure:
        """Create a comprehensive dashboard with key metrics."""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Methods by Category',
                'Abstraction Levels',
                'Duplicate Groups',
                'Methods by Source (Top 10)',
                'Toolkit Sizes',
                'Key Metrics'
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "table"}]
            ]
        )

        # 1. Category pie
        if category_report:
            distribution = category_report['summary']['distribution']
            category_names = [list(category_report['by_category'].values())[i][0]['category_name']
                            if category_report['by_category'] else k
                            for i, k in enumerate(distribution.keys())]

            fig.add_trace(
                go.Pie(labels=category_names, values=list(distribution.values()), hole=0.3),
                row=1, col=1
            )

        # 2. Abstraction bar
        summary = abstraction_report['summary']
        fig.add_trace(
            go.Bar(
                x=['High', 'Medium', 'Low'],
                y=[summary['high_level_count'], summary['medium_level_count'], summary['low_level_count']],
                marker_color=['#3498db', '#f39c12', '#e74c3c']
            ),
            row=1, col=2
        )

        # 3. Duplicate groups indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=duplicate_report['summary']['duplicate_groups'],
                title={"text": "Duplicate Groups"},
                delta={'reference': 0}
            ),
            row=1, col=3
        )

        # 4. Source distribution (top 10)
        source_counts = {}
        for method in methods:
            source_counts[method.source] = source_counts.get(method.source, 0) + 1
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        fig.add_trace(
            go.Bar(
                x=[s[0][:20] for s in sorted_sources],  # Truncate names
                y=[s[1] for s in sorted_sources],
                marker_color='#27ae60'
            ),
            row=2, col=1
        )

        # 5. Toolkit sizes
        if toolkit_report and toolkit_report['toolkits']:
            toolkit_sizes = [t['size'] for t in toolkit_report['toolkits'][:10]]
            toolkit_ids = [f"T{t['toolkit_id']}" for t in toolkit_report['toolkits'][:10]]

            fig.add_trace(
                go.Bar(x=toolkit_ids, y=toolkit_sizes, marker_color='#9b59b6'),
                row=2, col=2
            )

        # 6. Key metrics table
        metrics_data = [
            ["Total Methods", len(methods)],
            ["Duplicate Groups", duplicate_report['summary']['duplicate_groups']],
            ["Reduction Potential", duplicate_report['summary']['reduction_potential']],
            ["High Compat Pairs", compatibility_report['summary']['high_compatibility_pairs']],
            ["Toolkits Generated", toolkit_report['summary']['total_toolkits'] if toolkit_report else 0],
            ["Categories Used", category_report['summary']['categories_count'] if category_report else 0]
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[[m[0] for m in metrics_data], [m[1] for m in metrics_data]])
            ),
            row=2, col=3
        )

        fig.update_layout(
            height=1000,
            title_text="Methods Analysis Dashboard",
            showlegend=False
        )

        return fig

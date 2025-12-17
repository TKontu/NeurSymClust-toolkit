"""
Intelligent category assignment based on multi-dimensional scores.
Replaces LLM-based categorization with algorithmic scoring.
"""
import numpy as np
from typing import Dict, List, Tuple


class IntelligentCategoryAssigner:
    """
    Assign methods to strategic categories based on 9D scores.
    """

    def __init__(self):
        """Initialize category definitions"""
        self.categories = {
            '⭐ Stars': {
                'description': 'High impact, easy to implement, fast results - the best of all worlds',
                'criteria': self._is_star
            },
            '⚡ Quick Wins': {
                'description': 'Fast time to value, easy implementation, decent impact',
                'criteria': self._is_quick_win
            },
            '🚀 Strategic Investments': {
                'description': 'High impact, broad applicability, worth the difficulty',
                'criteria': self._is_strategic_investment
            },
            '🏗️ Foundation Builders': {
                'description': 'Easy to adopt, universally applicable, build capability',
                'criteria': self._is_foundation_builder
            },
            '💎 High-Risk High-Reward': {
                'description': 'Very high impact but very difficult to implement',
                'criteria': self._is_high_risk_high_reward
            },
            '🎯 Niche Solutions': {
                'description': 'Narrow applicability, context-specific',
                'criteria': self._is_niche_solution
            },
            '🐢 Long-Term Investments': {
                'description': 'Slow time to value but high strategic impact',
                'criteria': self._is_long_term_investment
            },
            '📊 Standard Practices': {
                'description': 'Middle of the road - reliable but not exceptional',
                'criteria': self._is_standard_practice
            }
        }

    def _is_star(self, scores: Dict) -> Tuple[bool, float]:
        """Stars: High impact + easy implementation + fast value"""
        impact = scores['impact_potential']
        difficulty = scores['implementation_difficulty']
        time = scores['time_to_value']

        # High impact, low difficulty, fast time
        if impact >= 70 and difficulty <= 30 and time >= 70:
            confidence = np.mean([impact, 100 - difficulty, time]) / 100
            return True, confidence

        return False, 0.0

    def _is_quick_win(self, scores: Dict) -> Tuple[bool, float]:
        """Quick Wins: Fast + easy + decent impact"""
        impact = scores['impact_potential']
        difficulty = scores['implementation_difficulty']
        time = scores['time_to_value']

        # Fast and easy, moderate impact
        if time >= 70 and difficulty <= 40 and impact >= 50:
            confidence = (time * 0.4 + (100 - difficulty) * 0.35 + impact * 0.25) / 100
            return True, confidence

        return False, 0.0

    def _is_strategic_investment(self, scores: Dict) -> Tuple[bool, float]:
        """Strategic Investments: High impact + broad applicability"""
        impact = scores['impact_potential']
        applicability = scores['applicability']
        scope = scores['scope']

        # High impact and broad reach, even if difficult
        if impact >= 70 and applicability >= 60 and scope >= 60:
            confidence = (impact * 0.45 + applicability * 0.35 + scope * 0.2) / 100
            return True, confidence

        return False, 0.0

    def _is_foundation_builder(self, scores: Dict) -> Tuple[bool, float]:
        """Foundation Builders: Easy + broad applicability"""
        difficulty = scores['implementation_difficulty']
        applicability = scores['applicability']
        ease_adoption = scores['ease_adoption']

        # Easy and universally applicable
        if difficulty <= 35 and applicability >= 70 and ease_adoption >= 65:
            confidence = ((100 - difficulty) * 0.35 + applicability * 0.35 + ease_adoption * 0.3) / 100
            return True, confidence

        return False, 0.0

    def _is_high_risk_high_reward(self, scores: Dict) -> Tuple[bool, float]:
        """High-Risk High-Reward: Very high impact + very difficult"""
        impact = scores['impact_potential']
        difficulty = scores['implementation_difficulty']

        # Very high impact but very hard
        if impact >= 80 and difficulty >= 75:
            confidence = (impact * 0.6 + difficulty * 0.4) / 100
            return True, confidence

        return False, 0.0

    def _is_niche_solution(self, scores: Dict) -> Tuple[bool, float]:
        """Niche Solutions: Narrow applicability"""
        applicability = scores['applicability']

        # Narrow applicability
        if applicability <= 35:
            confidence = (100 - applicability) / 100
            return True, confidence

        return False, 0.0

    def _is_long_term_investment(self, scores: Dict) -> Tuple[bool, float]:
        """Long-Term Investments: Slow but high strategic impact"""
        impact = scores['impact_potential']
        time = scores['time_to_value']
        scope = scores['scope']
        temporality = scores['temporality']

        # Slow time to value but high strategic impact
        if time <= 35 and impact >= 65 and (scope >= 65 or temporality >= 65):
            confidence = (impact * 0.4 + scope * 0.2 + temporality * 0.2 + (100 - time) * 0.2) / 100
            return True, confidence

        return False, 0.0

    def _is_standard_practice(self, scores: Dict) -> Tuple[bool, float]:
        """Standard Practices: Middle of the road (fallback category)"""
        # This is the default - returns low confidence always
        return True, 0.3

    def assign_category(self, scores: Dict) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Assign a method to the best-fit category.

        Args:
            scores: Dictionary with all 9D scores

        Returns:
            (category_name, confidence, all_matches)
        """
        matches = []

        # Check each category
        for cat_name, cat_info in self.categories.items():
            is_match, confidence = cat_info['criteria'](scores)
            if is_match:
                matches.append((cat_name, confidence))

        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return best match (or Standard Practices as fallback)
        if matches and matches[0][1] > 0.35:  # Confidence threshold
            return matches[0][0], matches[0][1], matches
        else:
            return '📊 Standard Practices', 0.5, matches

    def calculate_grades(self, scores: Dict) -> Dict[str, float]:
        """
        Calculate various grading metrics.

        Returns:
            Dictionary with different grade perspectives
        """
        # Simple average
        simple_avg = np.mean([
            scores['ease_adoption'],
            100 - scores['implementation_difficulty'],
            scores['impact_potential'],
            scores['time_to_value'],
            scores['applicability']
        ])

        # ROI-focused (high impact, low difficulty)
        roi_grade = (
            scores['impact_potential'] * 0.4 +
            (100 - scores['implementation_difficulty']) * 0.3 +
            scores['time_to_value'] * 0.2 +
            scores['applicability'] * 0.1
        )

        # Quick wins grade (fast + easy + impact)
        quick_wins_grade = (
            scores['time_to_value'] * 0.35 +
            scores['ease_adoption'] * 0.35 +
            scores['impact_potential'] * 0.2 +
            scores['applicability'] * 0.1
        )

        # Strategic grade (high impact + broad + strategic scope)
        strategic_grade = (
            scores['impact_potential'] * 0.45 +
            scores['applicability'] * 0.35 +
            scores['scope'] * 0.2
        )

        # Efficiency score (impact per unit difficulty)
        efficiency_grade = (
            scores['impact_potential'] * (100 - scores['implementation_difficulty']) / 100
        )

        # Composite score with multipliers
        composite_grade = (
            (scores['impact_potential'] * 0.4 + scores['applicability'] * 0.2) *
            (1 + (100 - scores['implementation_difficulty']) / 200) *
            (1 + scores['time_to_value'] / 200)
        )

        return {
            'simple_average': simple_avg,
            'roi_grade': roi_grade,
            'quick_wins_grade': quick_wins_grade,
            'strategic_grade': strategic_grade,
            'efficiency_grade': efficiency_grade,
            'composite_grade': composite_grade
        }

    def assign_letter_grade(self, composite_score: float) -> str:
        """Assign letter grade based on composite score"""
        if composite_score >= 90:
            return 'A+'
        elif composite_score >= 85:
            return 'A'
        elif composite_score >= 80:
            return 'A-'
        elif composite_score >= 75:
            return 'B+'
        elif composite_score >= 70:
            return 'B'
        elif composite_score >= 65:
            return 'B-'
        elif composite_score >= 60:
            return 'C+'
        elif composite_score >= 55:
            return 'C'
        elif composite_score >= 50:
            return 'C-'
        elif composite_score >= 45:
            return 'D+'
        elif composite_score >= 40:
            return 'D'
        else:
            return 'F'

    def calculate_percentiles(self, all_scores: List[Dict]) -> Dict[int, Dict[str, float]]:
        """
        Calculate percentile ranks for each method across all grades.

        Args:
            all_scores: List of all method scores

        Returns:
            Dictionary mapping method index to percentile ranks
        """
        # Extract grade values for each method
        grade_values = {
            'roi_grade': [],
            'quick_wins_grade': [],
            'strategic_grade': [],
            'composite_grade': []
        }

        indices = []
        for score in all_scores:
            indices.append(score.get('index', 0))
            grades = self.calculate_grades(score)
            for grade_type in grade_values:
                grade_values[grade_type].append(grades[grade_type])

        # Calculate percentiles
        percentiles = {}
        for i, idx in enumerate(indices):
            percentiles[idx] = {}
            for grade_type, values in grade_values.items():
                # Percentile rank
                value = values[i]
                percentile = (sum(1 for v in values if v < value) / len(values)) * 100
                percentiles[idx][f'{grade_type}_percentile'] = percentile

        return percentiles

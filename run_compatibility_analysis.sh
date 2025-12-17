#!/bin/bash
# Quick-start script for compatibility analysis

set -e  # Exit on error

echo "=================================="
echo "Method Compatibility Analysis"
echo "=================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please create it first: python3 -m venv venv"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Parse arguments
MAX_PAIRS=2000
MODE="recommended"

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MAX_PAIRS=50
            MODE="test"
            shift
            ;;
        --small)
            MAX_PAIRS=500
            MODE="small"
            shift
            ;;
        --recommended)
            MAX_PAIRS=2000
            MODE="recommended"
            shift
            ;;
        --large)
            MAX_PAIRS=5000
            MODE="large"
            shift
            ;;
        --xlarge)
            MAX_PAIRS=10000
            MODE="xlarge"
            shift
            ;;
        --comprehensive)
            MAX_PAIRS=20000
            MODE="comprehensive"
            shift
            ;;
        --max-pairs)
            MAX_PAIRS="$2"
            MODE="custom"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test|--small|--recommended|--large|--xlarge|--comprehensive|--max-pairs N]"
            echo ""
            echo "With 595 methods, total possible pairs: 176,715"
            echo "Modes use strategic sampling to focus on most important relationships"
            exit 1
            ;;
    esac
done

echo "Mode: $MODE ($MAX_PAIRS pairs)"
echo ""

# Run compatibility analysis
echo "Running compatibility analysis..."
echo "  Analyzing $MAX_PAIRS method pairs"
echo "  Estimated time: ~$((MAX_PAIRS * 2 / 100 / 60))m $((MAX_PAIRS * 2 / 100 % 60))s (with 100 concurrent requests)"
echo ""

python analyze_compatibility.py --max-pairs $MAX_PAIRS

if [ $? -ne 0 ]; then
    echo "❌ Analysis failed. Check logs above for errors."
    exit 1
fi

echo ""
echo "=================================="
echo "✅ Analysis Complete!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  📊 results/compatibility_analysis.json  - Full analysis results"
echo "  🕸️  results/compatibility_graph.gml     - Network graph (GML format)"
echo "  💾 results/compatibility_checkpoint.pkl - Checkpoint for incremental analysis"
echo ""
echo "To generate visualizations:"
echo "  python visualize_compatibility.py"
echo ""
echo "Or explore JSON:"
echo "  cat results/compatibility_analysis.json | jq '.statistics'"
echo "  cat results/compatibility_analysis.json | jq '.metadata'"
echo ""

#!/bin/bash
# Incremental compatibility analysis - runs batches until complete
# Analyzes all pairs in manageable chunks to avoid data loss

set -e  # Exit on error

BATCH_SIZE=10000
TOTAL_POSSIBLE_PAIRS=176715  # For 595 methods: 595 * 594 / 2

echo "=========================================="
echo "Full Compatibility Analysis (Incremental)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Batch size: $BATCH_SIZE pairs per run"
echo "  Total possible pairs: $TOTAL_POSSIBLE_PAIRS"
echo "  Strategy: Incremental batches with checkpointing"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please create it first: python3 -m venv venv"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Function to get current progress
get_analyzed_count() {
    if [ -f "results/compatibility_checkpoint.pkl" ]; then
        python3 -c "
import pickle
from pathlib import Path
checkpoint_file = Path('results/compatibility_checkpoint.pkl')
if checkpoint_file.exists():
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    print(len(checkpoint.get('analyzed_pairs', set())))
else:
    print(0)
" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Get starting count
ANALYZED_PAIRS=$(get_analyzed_count)
echo "Current progress: $ANALYZED_PAIRS / $TOTAL_POSSIBLE_PAIRS pairs analyzed"
echo ""

if [ $ANALYZED_PAIRS -ge $TOTAL_POSSIBLE_PAIRS ]; then
    echo "✅ Analysis already complete! All pairs analyzed."
    echo ""
    echo "To regenerate reports:"
    echo "  python analyze_compatibility.py --report-only"
    exit 0
fi

# Calculate estimated batches
REMAINING=$((TOTAL_POSSIBLE_PAIRS - ANALYZED_PAIRS))
ESTIMATED_BATCHES=$(( (REMAINING + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Estimated batches needed: $ESTIMATED_BATCHES"
echo "Estimated total time: ~$((ESTIMATED_BATCHES * BATCH_SIZE * 2 / 100 / 60)) minutes (at 100 concurrent requests)"
echo ""
read -p "Start incremental analysis? [Y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Aborted."
    exit 0
fi
echo ""

# Run batches
BATCH_NUM=1
START_TIME=$(date +%s)

while true; do
    # Get current count
    ANALYZED_PAIRS=$(get_analyzed_count)
    REMAINING=$((TOTAL_POSSIBLE_PAIRS - ANALYZED_PAIRS))

    if [ $REMAINING -le 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ Analysis Complete!"
        echo "=========================================="
        echo ""
        echo "Building compatibility graph and generating reports..."
        echo ""

        # Build graph and generate final reports
        python analyze_compatibility.py --report-only --build-graph

        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Graph and reports generated successfully!"
        else
            echo ""
            echo "⚠️ Graph building failed. You can try again manually:"
            echo "   python analyze_compatibility.py --report-only --build-graph"
        fi

        break
    fi

    # Determine batch size (use smaller size for last batch)
    CURRENT_BATCH_SIZE=$BATCH_SIZE
    if [ $REMAINING -lt $BATCH_SIZE ]; then
        CURRENT_BATCH_SIZE=$REMAINING
    fi

    PROGRESS_PCT=$((ANALYZED_PAIRS * 100 / TOTAL_POSSIBLE_PAIRS))

    echo "=========================================="
    echo "Batch #$BATCH_NUM"
    echo "=========================================="
    echo "Progress: $ANALYZED_PAIRS / $TOTAL_POSSIBLE_PAIRS pairs ($PROGRESS_PCT%)"
    echo "Remaining: $REMAINING pairs"
    echo "This batch: $CURRENT_BATCH_SIZE pairs"
    echo ""

    # Run analysis with exhaustive mode for systematic coverage
    python analyze_compatibility.py --max-pairs $CURRENT_BATCH_SIZE --exhaustive

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Batch #$BATCH_NUM failed!"
        echo ""
        echo "Progress saved in checkpoint. You can resume by running this script again."
        echo "Current progress: $ANALYZED_PAIRS pairs analyzed"
        exit 1
    fi

    # Show batch completion
    NEW_ANALYZED=$(get_analyzed_count)
    ACTUALLY_ADDED=$((NEW_ANALYZED - ANALYZED_PAIRS))

    echo ""
    echo "✅ Batch #$BATCH_NUM complete: +$ACTUALLY_ADDED pairs"
    echo "   Total analyzed: $NEW_ANALYZED / $TOTAL_POSSIBLE_PAIRS"
    echo ""

    # Update counter
    BATCH_NUM=$((BATCH_NUM + 1))

    # Small delay between batches
    sleep 2
done

# Calculate total time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "📊 ANALYSIS SUMMARY"
echo "=========================================="
echo "Total batches run: $((BATCH_NUM - 1))"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "✅ Results saved to:"
echo "  📊 results/compatibility_analysis.json  - Full analysis results"
echo "  🕸️  results/compatibility_graph.gml     - Network graph"
echo "  💾 results/compatibility_checkpoint.pkl - Analysis checkpoint"
echo ""
echo "📈 Next steps:"
echo "  python visualize_compatibility.py      - Generate visualizations"
echo "  python build_method_portfolios.py      - Build method toolkits"
echo ""
echo "🔍 Explore results:"
echo "  cat results/compatibility_analysis.json | jq '.statistics'"
echo "  cat results/compatibility_analysis.json | jq '.metadata'"
echo ""

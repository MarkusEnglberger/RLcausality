#!/bin/bash
# Find and patch Phi-3 modeling file to use new DynamicCache API

# Load Python module
module load Python/3.10.4-GCCcore-11.3.0

# Activate virtual environment
source venv/bin/activate

echo "======================================"
echo "Finding Phi-3 modeling file"
echo "======================================"

# Find the modeling_phi3.py file in the cache
MODELING_FILE=$(find data/cache -name "modeling_phi3.py" 2>/dev/null | head -n 1)

if [ -z "$MODELING_FILE" ]; then
    echo "Error: Could not find modeling_phi3.py"
    echo "Searching in transformers installation..."
    TRANSFORMERS_PATH=$(python -c "import transformers; print(transformers.__file__.replace('__init__.py', ''))")
    MODELING_FILE=$(find "$TRANSFORMERS_PATH" -name "modeling_phi3.py" 2>/dev/null | head -n 1)
fi

if [ -z "$MODELING_FILE" ]; then
    echo "Error: Could not find modeling_phi3.py anywhere"
    exit 1
fi

echo "Found: $MODELING_FILE"
echo ""

# Check if file has the problematic line
echo "Checking for problematic code..."
if grep -q "past_key_values.seen_tokens" "$MODELING_FILE"; then
    echo "✓ Found problematic code using past_key_values.seen_tokens"
    echo ""
    echo "Creating backup..."
    cp "$MODELING_FILE" "${MODELING_FILE}.backup"
    echo "Backup created: ${MODELING_FILE}.backup"
    echo ""

    echo "Patching file..."
    # Replace seen_tokens with get_seq_length()
    sed -i 's/past_key_values\.seen_tokens/past_key_values.get_seq_length()/g' "$MODELING_FILE"

    echo "✓ Patching complete!"
    echo ""
    echo "Verifying changes..."
    if grep -q "past_key_values.get_seq_length()" "$MODELING_FILE"; then
        echo "✓ Patch applied successfully"
        echo ""
        echo "Changed lines:"
        grep -n "past_key_values.get_seq_length()" "$MODELING_FILE"
    else
        echo "✗ Patch may have failed"
        exit 1
    fi
else
    echo "✗ Could not find the problematic code"
    echo "The file may already be patched or have different code"
fi

echo ""
echo "======================================"
echo "Done! You can now retry GRPO training."
echo "======================================"
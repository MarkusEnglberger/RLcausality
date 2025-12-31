#!/bin/bash
# Patch the specific Phi-3 modeling file to fix DynamicCache API issue

# The exact file path from the error
MODELING_FILE="/vast.mnt/home/20250638/RLcausality/data/cache/modules/transformers_modules/microsoft/Phi_hyphen_3_dot_5_hyphen_mini_hyphen_instruct/3145e03a9fd4cdd7cd953c34d9bbf7ad606122ca/modeling_phi3.py"

echo "======================================"
echo "Patching Phi-3 modeling file"
echo "======================================"
echo "File: $MODELING_FILE"
echo ""

# Check if file exists
if [ ! -f "$MODELING_FILE" ]; then
    echo "Error: File not found!"
    exit 1
fi

# Check if file has the problematic line
echo "Checking for problematic code..."
if grep -q "past_key_values.seen_tokens" "$MODELING_FILE"; then
    echo "✓ Found problematic code using past_key_values.seen_tokens"

    # Show the problematic line
    echo ""
    echo "Problematic line (before):"
    grep -n "past_key_values.seen_tokens" "$MODELING_FILE"

    echo ""
    echo "Creating backup..."
    cp "$MODELING_FILE" "${MODELING_FILE}.backup"
    echo "✓ Backup created: ${MODELING_FILE}.backup"

    echo ""
    echo "Applying patch..."
    # Replace seen_tokens with get_seq_length()
    sed -i 's/past_key_values\.seen_tokens/past_key_values.get_seq_length()/g' "$MODELING_FILE"

    echo "✓ Patch applied!"

    echo ""
    echo "Verifying changes..."
    if grep -q "past_key_values.get_seq_length()" "$MODELING_FILE"; then
        echo "✓ Verification successful!"
        echo ""
        echo "Patched line (after):"
        grep -n "past_key_values.get_seq_length()" "$MODELING_FILE"
    else
        echo "✗ Verification failed! Restoring backup..."
        cp "${MODELING_FILE}.backup" "$MODELING_FILE"
        exit 1
    fi
else
    echo "✗ Could not find the problematic code."
    echo "The file may already be patched."
    echo ""
    echo "Checking for get_seq_length()..."
    if grep -q "past_key_values.get_seq_length()" "$MODELING_FILE"; then
        echo "✓ File appears to be already patched!"
    else
        echo "✗ File doesn't have expected code. Manual inspection needed."
        echo ""
        echo "Showing context around line 1298:"
        sed -n '1295,1305p' "$MODELING_FILE"
    fi
fi

echo ""
echo "======================================"
echo "Done!"
echo "======================================"
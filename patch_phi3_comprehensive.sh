#!/bin/bash
# Comprehensive patch for Phi-3 modeling file DynamicCache API issues

MODELING_FILE="/vast.mnt/home/20250638/RLcausality/data/cache/modules/transformers_modules/microsoft/Phi_hyphen_3_dot_5_hyphen_mini_hyphen_instruct/3145e03a9fd4cdd7cd953c34d9bbf7ad606122ca/modeling_phi3.py"

echo "======================================"
echo "Comprehensive Phi-3 Modeling Patch"
echo "======================================"
echo "Target file: $MODELING_FILE"
echo ""

# Check if file exists
if [ ! -f "$MODELING_FILE" ]; then
    echo "Error: File not found at expected location!"
    echo "Searching for modeling_phi3.py in cache..."
    FOUND_FILE=$(find /vast.mnt/home/20250638/RLcausality/data/cache -name "modeling_phi3.py" 2>/dev/null | head -n 1)
    if [ -n "$FOUND_FILE" ]; then
        echo "Found at: $FOUND_FILE"
        MODELING_FILE="$FOUND_FILE"
    else
        echo "Could not find modeling_phi3.py anywhere!"
        exit 1
    fi
fi

echo "Creating backup..."
cp "$MODELING_FILE" "${MODELING_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
echo "✓ Backup created"
echo ""

# Show the problematic section before patching
echo "Code before patching (lines 1295-1305):"
echo "----------------------------------------"
sed -n '1295,1305p' "$MODELING_FILE"
echo "----------------------------------------"
echo ""

# Apply all necessary patches
echo "Applying patches..."

# Patch 1: Replace seen_tokens with get_seq_length()
if grep -q "past_key_values\.seen_tokens" "$MODELING_FILE"; then
    echo "✓ Patching seen_tokens → get_seq_length()"
    sed -i 's/past_key_values\.seen_tokens/past_key_values.get_seq_length()/g' "$MODELING_FILE"
fi

# Patch 2: Replace get_max_length() with get_max_cache_shape()
if grep -q "past_key_values\.get_max_length()" "$MODELING_FILE"; then
    echo "✓ Patching get_max_length() → get_max_cache_shape()"
    sed -i 's/past_key_values\.get_max_length()/past_key_values.get_max_cache_shape()/g' "$MODELING_FILE"
fi

# Patch 3: Handle max_cache_len attribute
if grep -q "past_key_values\.max_cache_len" "$MODELING_FILE"; then
    echo "✓ Patching max_cache_len access"
    # This one is trickier - need to check context
    sed -i 's/past_key_values\.max_cache_len/past_key_values.get_max_cache_shape()/g' "$MODELING_FILE"
fi

echo ""
echo "Code after patching (lines 1295-1305):"
echo "----------------------------------------"
sed -n '1295,1305p' "$MODELING_FILE"
echo "----------------------------------------"
echo ""

# Verify patches
echo "Verifying patches..."
ERRORS=0

if grep -q "past_key_values\.seen_tokens" "$MODELING_FILE"; then
    echo "✗ Warning: seen_tokens still present"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "past_key_values\.get_max_length()" "$MODELING_FILE"; then
    echo "✗ Warning: get_max_length() still present"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -eq 0 ]; then
    echo "✓ All patches applied successfully!"
else
    echo "⚠ Some issues may remain. Check the output above."
fi

echo ""
echo "======================================"
echo "Patch complete!"
echo "======================================"
echo ""
echo "Note: The model may re-download this file on next run."
echo "To prevent this, we need to pin the model revision in the config."

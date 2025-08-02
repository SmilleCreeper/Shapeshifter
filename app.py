import streamlit as st
import numpy as np
from PIL import Image
import io
import colorsys

st.title("🧠 Sparse HSL Rectangular Encoder")
st.write("Upload a PNG image and compare standard RGB size vs. optimized HSL rectangular format.")

# --- Upload image ---
uploaded_file = st.file_uploader("Upload PNG Image", type=["png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape

    st.image(image, caption=f"Original Image ({w}x{h})", use_column_width=True)

    # --- Convert RGB to HSL ---
    st.subheader("🎨 RGB to HSL Conversion")
    progress_conversion = st.progress(0)
    conversion_text = st.empty()
    
    # Convert RGB to HSL
    hsl_img = np.zeros_like(img_np, dtype=np.float32)
    total_pixels = h * w
    
    for y in range(h):
        for x in range(w):
            r, g, b = img_np[y, x] / 255.0  # Normalize to 0-1
            hsl_h, hsl_l, hsl_s = colorsys.rgb_to_hls(r, g, b)  # Note: HLS order in colorsys
            hsl_img[y, x] = [hsl_h * 360, hsl_s * 100, hsl_l * 100]  # Scale to H:0-360, S:0-100, L:0-100
            
            if (y * w + x) % 1000 == 0:  # Update progress every 1000 pixels
                progress_conversion.progress((y * w + x) / total_pixels)
                conversion_text.text(f"Converting pixel {y * w + x}/{total_pixels}")
    
    progress_conversion.progress(1.0)
    conversion_text.text("HSL conversion complete!")
    
    # Quantize HSL to integers for compression
    H = np.round(hsl_img[:, :, 0]).astype(np.uint16)  # Hue: 0-360
    S = np.round(hsl_img[:, :, 1]).astype(np.uint8)   # Saturation: 0-100  
    L = np.round(hsl_img[:, :, 2]).astype(np.uint8)   # Lightness: 0-100

    # --- Extract per-channel unique values ---
    unique_H = np.unique(H)
    unique_S = np.unique(S)
    unique_L = np.unique(L)

    st.markdown(f"**HSL Channel Analysis:**")
    st.markdown(f"- Hue (H): {len(unique_H)} unique values (range: {unique_H.min()}-{unique_H.max()})")
    st.markdown(f"- Saturation (S): {len(unique_S)} unique values (range: {unique_S.min()}-{unique_S.max()})")
    st.markdown(f"- Lightness (L): {len(unique_L)} unique values (range: {unique_L.min()}-{unique_L.max()})")

    channel_maps = {
        'H': {val: idx for idx, val in enumerate(unique_H)},
        'S': {val: idx for idx, val in enumerate(unique_S)},
        'L': {val: idx for idx, val in enumerate(unique_L)},
    }

    H_ids = np.vectorize(channel_maps['H'].get)(H)
    S_ids = np.vectorize(channel_maps['S'].get)(S)
    L_ids = np.vectorize(channel_maps['L'].get)(L)

    st.subheader("🚀 Encoding Progress")
    progress_bar = st.progress(0)
    progress_text = st.empty()

    debug_table = {}

    def calculate_optimal_bits(value):
        """Calculate optimal bits needed using offset encoding, 0 uses 0 bits (implicit)"""
        if value == 0:
            return 0  # Zero uses no bits - implicit default
        elif 1 <= value <= 2:
            return 1  # 1 bit can represent 1-2
        elif 3 <= value <= 6:
            return 2  # 2 bits can represent 3-6  
        elif 7 <= value <= 14:
            return 3  # 3 bits can represent 7-14
        elif 15 <= value <= 30:
            return 4  # 4 bits can represent 15-30
        elif 31 <= value <= 62:
            return 5  # 5 bits can represent 31-62
        elif 63 <= value <= 126:
            return 6  # 6 bits can represent 63-126
        elif 127 <= value <= 254:
            return 7  # 7 bits can represent 127-254
        else:  # value == 255
            return 8  # 8 bits can represent 255

    def find_rectangles(channel_data, val):
        """Find rectangular regions of the same value"""
        h, w = channel_data.shape
        visited = np.zeros_like(channel_data, dtype=bool)
        rectangles = []
        
        for y in range(h):
            for x in range(w):
                if channel_data[y, x] == val and not visited[y, x]:
                    # Find the largest rectangle starting at (x, y)
                    # First, find the width of the rectangle at this row
                    width = 0
                    while x + width < w and channel_data[y, x + width] == val and not visited[y, x + width]:
                        width += 1
                    
                    # Now find the height - how many rows can we extend with this width
                    height = 1
                    while y + height < h:
                        # Check if the entire width can be extended to this row
                        can_extend = True
                        for dx in range(width):
                            if channel_data[y + height, x + dx] != val or visited[y + height, x + dx]:
                                can_extend = False
                                break
                        if not can_extend:
                            break
                        height += 1
                    
                    # Mark this rectangle as visited
                    for dy in range(height):
                        for dx in range(width):
                            visited[y + dy, x + dx] = True
                    
                    rectangles.append((x, y, width, height))
        
        return rectangles

    def encode_channel_stream(channel_data, channel_name, unique_values):
        # Calculate bits needed for each individual value in lookup table using offset encoding
        value_bits_map = {}
        lookup_table_bits = 0
        for val in unique_values:
            bits_needed = calculate_optimal_bits(val)
            value_bits_map[val] = bits_needed
            lookup_table_bits += bits_needed
        compressed_bits = lookup_table_bits
        
        h, w = channel_data.shape
        rect_table = {}
        total = len(unique_values)
        channel_debug = {}
        
        for idx, val in enumerate(unique_values):
            progress_text.text(f"Encoding {channel_name} value {idx + 1}/{total}")
            rectangles = find_rectangles(channel_data, idx)  # Use idx because channel_data contains mapped indices
            rect_table[val] = rectangles
            
            if rectangles:
                # Calculate bits needed for each individual rectangle
                rect_bits = 0
                rect_details = []
                last_x, last_y = 0, 0  # Track last position for relative coordinates
                last_w, last_h = 0, 0  # Track last size for differential encoding
                
                for i, (rect_x, rect_y, rect_w, rect_h) in enumerate(rectangles):
                    # Use relative positioning from last rectangle
                    rel_x = rect_x - last_x
                    rel_y = rect_y - last_y
                    
                    # Use differential encoding for size (0 means "same as previous")
                    if i == 0:  # First rectangle must encode size explicitly
                        delta_w = rect_w
                        delta_h = rect_h
                    else:
                        # If size matches previous, encode as 0 (which uses 0 bits)
                        delta_w = 0 if rect_w == last_w else rect_w
                        delta_h = 0 if rect_h == last_h else rect_h
                    
                    x_bits = calculate_optimal_bits(rel_x)
                    y_bits = calculate_optimal_bits(rel_y)
                    w_bits = calculate_optimal_bits(delta_w)
                    h_bits = calculate_optimal_bits(delta_h)
                    
                    rect_bits += x_bits + y_bits + w_bits + h_bits
                    
                    # Track what was actually encoded vs what the rectangle size is
                    rect_details.append({
                        'rel_x': rel_x, 'rel_y': rel_y, 
                        'actual_width': rect_w, 'actual_height': rect_h,
                        'encoded_width': delta_w, 'encoded_height': delta_h,
                        'x_bits': x_bits, 'y_bits': y_bits, 'w_bits': w_bits, 'h_bits': h_bits,
                        'total_bits': x_bits + y_bits + w_bits + h_bits,
                        'inherited_w': delta_w == 0 and i > 0,
                        'inherited_h': delta_h == 0 and i > 0
                    })
                    
                    # Update last position to end of this rectangle
                    last_x = rect_x + rect_w
                    last_y = rect_y
                    # Update last size for next iteration
                    last_w = rect_w
                    last_h = rect_h
                
                compressed_bits += rect_bits
                
                channel_debug[val] = {
                    'rectangles': len(rectangles),
                    'rectangle_coords': rectangles,  # Store actual rectangle coordinates
                    'rect_bits': rect_bits,
                    'rect_details': rect_details
                }
            else:
                channel_debug[val] = {
                    'rectangles': 0,
                    'rectangle_coords': [],
                    'rect_bits': 0,
                    'rect_details': []
                }
            
            # Calculate progress based on HSL channels
            channel_index = {'H': 0, 'S': 1, 'L': 2}[channel_name]
            progress_bar.progress(((idx + 1) + total * channel_index) / (3 * total))
        
        # Add lookup table info to debug
        channel_debug['_lookup_table'] = {
            'unique_values': len(unique_values),
            'value_bits_map': value_bits_map,
            'lookup_bits': lookup_table_bits
        }
        debug_table[channel_name] = channel_debug
        return compressed_bits

    bits_H = encode_channel_stream(H_ids, 'H', unique_H)
    bits_S = encode_channel_stream(S_ids, 'S', unique_S)
    bits_L = encode_channel_stream(L_ids, 'L', unique_L)

    compressed_bits = bits_H + bits_S + bits_L
    compressed_bytes = (compressed_bits + 7) // 8

    # --- Output Comparison ---
    original_bytes = h * w * 3  # 3 bytes per pixel RGB

    st.subheader("📊 Compression Results")
    st.markdown(f"**Original RGB size:** {original_bytes} bytes")
    st.markdown(f"**Compressed (HSL rectangular format):** {compressed_bytes} bytes")
    st.markdown(f"**Compression ratio:** {compressed_bytes/original_bytes:.2%}")

    if len(unique_H) > 361 or len(unique_S) > 101 or len(unique_L) > 101:
        st.warning("One or more HSL channels have unexpected value ranges.")
    else:
        st.success("Compression completed using per-channel HSL rectangular encoding.")

    st.subheader("🛠️ Compression Analysis")
    
    # Create a comprehensive data table
    analysis_data = []
    
    for ch in ['H', 'S', 'L']:
        lookup_info = debug_table[ch]['_lookup_table']
        
        # Add lookup table row
        analysis_data.append({
            'Channel': ch,
            'Type': 'Lookup Table',
            'Values': lookup_info['unique_values'],
            'Total Bits': lookup_info['lookup_bits'],
            'Details': f"Unique values: {', '.join([f'{val}({bits}b)' for val, bits in lookup_info['value_bits_map'].items()][:5])}{'...' if len(lookup_info['value_bits_map']) > 5 else ''}"
        })
        
        # Add rectangle data for each value
        total_rect_bits = 0
        total_rectangles = 0
        for val, d in debug_table[ch].items():
            if val != '_lookup_table':
                total_rect_bits += d['rect_bits']
                total_rectangles += d['rectangles']
                
                # Create summary of savings for this value
                savings_summary = []
                zero_saves = 0
                inherited_saves = 0
                
                for rect in d['rect_details']:
                    if rect['x_bits'] == 0: zero_saves += 1
                    if rect['y_bits'] == 0: zero_saves += 1
                    if rect['w_bits'] == 0 and not rect['inherited_w']: zero_saves += 1
                    if rect['h_bits'] == 0 and not rect['inherited_h']: zero_saves += 1
                    if rect['inherited_w']: inherited_saves += 1
                    if rect['inherited_h']: inherited_saves += 1
                
                if zero_saves > 0: savings_summary.append(f"{zero_saves} zeros")
                if inherited_saves > 0: savings_summary.append(f"{inherited_saves} inherited")
                
                analysis_data.append({
                    'Channel': '',
                    'Type': f'Value {val}',
                    'Values': d['rectangles'],
                    'Total Bits': d['rect_bits'],
                    'Details': f"Rectangles: {d['rectangles']}, Savings: {', '.join(savings_summary) if savings_summary else 'None'}"
                })
        
        # Add channel summary
        analysis_data.append({
            'Channel': f'{ch} Total',
            'Type': 'Summary',
            'Values': total_rectangles,
            'Total Bits': lookup_info['lookup_bits'] + total_rect_bits,
            'Details': f"{lookup_info['lookup_bits']} lookup + {total_rect_bits} rectangles"
        })
        
        # Add separator
        analysis_data.append({
            'Channel': '─────',
            'Type': '─────────',
            'Values': '─────',
            'Total Bits': '─────',
            'Details': '─────────────────────────────────────────'
        })
    
    # Remove last separator
    analysis_data = analysis_data[:-1]
    
    # Display as dataframe
    import pandas as pd
    df = pd.DataFrame(analysis_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("H Channel", f"{debug_table['H']['_lookup_table']['lookup_bits'] + sum(d['rect_bits'] for k, d in debug_table['H'].items() if k != '_lookup_table')} bits")
    with col2:
        st.metric("S Channel", f"{debug_table['S']['_lookup_table']['lookup_bits'] + sum(d['rect_bits'] for k, d in debug_table['S'].items() if k != '_lookup_table')} bits")
    with col3:
        st.metric("L Channel", f"{debug_table['L']['_lookup_table']['lookup_bits'] + sum(d['rect_bits'] for k, d in debug_table['L'].items() if k != '_lookup_table')} bits")

    # Detailed Rectangle Analysis
    st.subheader("📋 Detailed Rectangle Analysis")
    
    # Create detailed rectangle data
    rect_data = []
    
    for ch in ['H', 'S', 'L']:
        for val, d in debug_table[ch].items():
            if val != '_lookup_table':
                for i, rect in enumerate(d['rect_details']):
                    # Calculate savings info
                    savings = []
                    if rect['x_bits'] == 0: savings.append('x=0')
                    if rect['y_bits'] == 0: savings.append('y=0')
                    if rect['inherited_w']: savings.append(f'w=inherited')
                    elif rect['w_bits'] == 0: savings.append('w=0')
                    if rect['inherited_h']: savings.append(f'h=inherited')  
                    elif rect['h_bits'] == 0: savings.append('h=0')
                    
                    rect_data.append({
                        'Channel': ch,
                        'Value': val,
                        'Rect #': i + 1,
                        'Rel X': rect['rel_x'],
                        'Rel Y': rect['rel_y'], 
                        'Width': rect['actual_width'],
                        'Height': rect['actual_height'],
                        'X Bits': rect['x_bits'],
                        'Y Bits': rect['y_bits'],
                        'W Bits': rect['w_bits'],
                        'H Bits': rect['h_bits'],
                        'Total Bits': rect['total_bits'],
                        'Savings': ', '.join(savings) if savings else 'None',
                        'Area': rect['actual_width'] * rect['actual_height'],
                        'Efficiency': f"{rect['total_bits'] / (rect['actual_width'] * rect['actual_height']):.3f}" if rect['actual_width'] * rect['actual_height'] > 0 else "0"
                    })
    
    # Display rectangle dataframe
    rect_df = pd.DataFrame(rect_data)
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        channel_filter = st.selectbox("Filter by Channel:", ['All'] + ['H', 'S', 'L'])
    with col2:
        min_area = st.number_input("Min Area:", min_value=0, value=0)
    with col3:
        show_savings_only = st.checkbox("Show only rectangles with savings")
    
    # Apply filters
    filtered_df = rect_df.copy()
    if channel_filter != 'All':
        filtered_df = filtered_df[filtered_df['Channel'] == channel_filter]
    if min_area > 0:
        filtered_df = filtered_df[filtered_df['Area'] >= min_area]
    if show_savings_only:
        filtered_df = filtered_df[filtered_df['Savings'] != 'None']
    
    st.dataframe(filtered_df, use_container_width=True, height=500)
    
    # Rectangle statistics
    if len(filtered_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rectangles", len(filtered_df))
        with col2:
            st.metric("Avg Bits/Rectangle", f"{filtered_df['Total Bits'].mean():.1f}")
        with col3:
            st.metric("Avg Area", f"{filtered_df['Area'].mean():.1f}")
        with col4:
            savings_count = len(filtered_df[filtered_df['Savings'] != 'None'])
            st.metric("Rectangles with Savings", f"{savings_count} ({100*savings_count/len(filtered_df):.1f}%)")
    else:
        st.info("No rectangles match the current filters.")

    # --- Image Reconstruction Test ---
    st.subheader("🔄 Test Reconstruction")
    st.write("Reconstruct the image from the compressed HSL rectangular format to verify the compression algorithm.")
    
    if st.button("🎯 Reconstruct Image", type="primary"):
        progress_reconstruction = st.progress(0)
        reconstruction_text = st.empty()
        
        # Initialize reconstructed HSL image
        reconstructed_hsl = np.zeros((h, w, 3), dtype=np.float32)
        
        def reconstruct_channel(channel_data, channel_name, unique_values, channel_index):
            """Reconstruct a channel from rectangle data"""
            reconstruction_text.text(f"Reconstructing {channel_name} channel...")
            
            for idx, val in enumerate(unique_values):
                if val in debug_table[channel_name] and 'rectangle_coords' in debug_table[channel_name][val]:
                    rectangles = debug_table[channel_name][val]['rectangle_coords']
                    
                    for rect_x, rect_y, rect_w, rect_h in rectangles:
                        # Fill rectangle with original HSL value (not the mapped index)
                        if channel_name == 'H':
                            reconstructed_hsl[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w, 0] = val
                        elif channel_name == 'S':
                            reconstructed_hsl[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w, 1] = val
                        else:  # L
                            reconstructed_hsl[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w, 2] = val
                
                progress_reconstruction.progress((idx + 1 + len(unique_values) * channel_index) / (3 * len(unique_values)))
        
        # Reconstruct each channel
        reconstruct_channel(H_ids, 'H', unique_H, 0)
        reconstruct_channel(S_ids, 'S', unique_S, 1)
        reconstruct_channel(L_ids, 'L', unique_L, 2)
        
        # Convert reconstructed HSL back to RGB
        reconstruction_text.text("Converting HSL back to RGB...")
        reconstructed_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                hsl_h = reconstructed_hsl[y, x, 0] / 360.0  # Convert back to 0-1
                hsl_s = reconstructed_hsl[y, x, 1] / 100.0  # Convert back to 0-1
                hsl_l = reconstructed_hsl[y, x, 2] / 100.0  # Convert back to 0-1
                
                # Convert HLS to RGB (note: colorsys uses HLS order)
                rgb_r, rgb_g, rgb_b = colorsys.hls_to_rgb(hsl_h, hsl_l, hsl_s)
                reconstructed_rgb[y, x] = [
                    int(round(rgb_r * 255)),
                    int(round(rgb_g * 255)),
                    int(round(rgb_b * 255))
                ]
            
            if y % 10 == 0:  # Update progress every 10 rows
                progress_reconstruction.progress(0.9 + 0.1 * y / h)
        
        progress_reconstruction.progress(1.0)
        reconstruction_text.text("Reconstruction complete!")
        
        # Convert original HSL back to RGB for comparison
        original_hsl_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                hsl_h = hsl_img[y, x, 0] / 360.0  # Convert back to 0-1
                hsl_s = hsl_img[y, x, 1] / 100.0  # Convert back to 0-1
                hsl_l = hsl_img[y, x, 2] / 100.0  # Convert back to 0-1
                
                # Convert HLS to RGB (note: colorsys uses HLS order)
                rgb_r, rgb_g, rgb_b = colorsys.hls_to_rgb(hsl_h, hsl_l, hsl_s)
                original_hsl_rgb[y, x] = [
                    int(round(rgb_r * 255)),
                    int(round(rgb_g * 255)),
                    int(round(rgb_b * 255))
                ]
        
        # Display all three images
        original_hsl_image = Image.fromarray(original_hsl_rgb)
        reconstructed_image = Image.fromarray(reconstructed_rgb)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original RGB", use_column_width=True)
        with col2:
            st.image(original_hsl_image, caption="Original → HSL → RGB", use_column_width=True)
        with col3:
            st.image(reconstructed_image, caption="Compressed → Reconstructed", use_column_width=True)
        
        # Calculate accuracy between all comparisons
        # Original RGB vs Original HSL conversion
        hsl_conversion_diff = np.abs(img_np.astype(np.int16) - original_hsl_rgb.astype(np.int16))
        hsl_max_diff = np.max(hsl_conversion_diff)
        hsl_mean_diff = np.mean(hsl_conversion_diff)
        hsl_perfect_pixels = np.sum(np.all(hsl_conversion_diff == 0, axis=2))
        
        # Original RGB vs Compressed reconstruction
        compressed_diff = np.abs(img_np.astype(np.int16) - reconstructed_rgb.astype(np.int16))
        compressed_max_diff = np.max(compressed_diff)
        compressed_mean_diff = np.mean(compressed_diff)
        compressed_perfect_pixels = np.sum(np.all(compressed_diff == 0, axis=2))
        
        # Original HSL vs Compressed HSL (compare quantized HSL with reconstructed HSL)
        # Create quantized HSL for fair comparison (this is what we actually encoded)
        quantized_hsl = np.zeros_like(hsl_img, dtype=np.float32)
        quantized_hsl[:, :, 0] = H.astype(np.float32)  # Use quantized H values
        quantized_hsl[:, :, 1] = S.astype(np.float32)  # Use quantized S values  
        quantized_hsl[:, :, 2] = L.astype(np.float32)  # Use quantized L values
        
        hsl_vs_compressed_hsl_diff = np.abs(quantized_hsl.astype(np.float64) - reconstructed_hsl.astype(np.float64))
        hsl_vs_compressed_hsl_max_diff = np.max(hsl_vs_compressed_hsl_diff)
        hsl_vs_compressed_hsl_mean_diff = np.mean(hsl_vs_compressed_hsl_diff)
        hsl_vs_compressed_hsl_perfect_pixels = np.sum(np.all(hsl_vs_compressed_hsl_diff == 0, axis=2))
        
        # Display comparison metrics
        st.subheader("📊 Reconstruction Accuracy Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original RGB vs HSL Conversion:**")
            st.metric("Max Difference", f"{hsl_max_diff}")
            st.metric("Mean Difference", f"{hsl_mean_diff:.2f}")
            st.metric("Perfect Pixels", f"{hsl_perfect_pixels}/{total_pixels}")
            hsl_accuracy = (hsl_perfect_pixels / total_pixels) * 100
            st.metric("Accuracy", f"{hsl_accuracy:.1f}%")
        
        with col2:
            st.markdown("**Original RGB vs Compressed:**")
            st.metric("Max Difference", f"{compressed_max_diff}")
            st.metric("Mean Difference", f"{compressed_mean_diff:.2f}")
            st.metric("Perfect Pixels", f"{compressed_perfect_pixels}/{total_pixels}")
            compressed_accuracy = (compressed_perfect_pixels / total_pixels) * 100
            st.metric("Accuracy", f"{compressed_accuracy:.1f}%")
        
        with col3:
            st.markdown("**Quantized HSL vs Compressed HSL:**")
            st.metric("Max Difference", f"{hsl_vs_compressed_hsl_max_diff:.2f}")
            st.metric("Mean Difference", f"{hsl_vs_compressed_hsl_mean_diff:.2f}")
            st.metric("Perfect Pixels", f"{hsl_vs_compressed_hsl_perfect_pixels}/{total_pixels}")
            hsl_vs_compressed_hsl_accuracy = (hsl_vs_compressed_hsl_perfect_pixels / total_pixels) * 100
            st.metric("Accuracy", f"{hsl_vs_compressed_hsl_accuracy:.1f}%")
        
        # Overall assessment
        st.subheader("🎯 Quality Assessment")
        
        if hsl_vs_compressed_hsl_max_diff == 0:
            st.success("🎉 Perfect compression! The rectangular encoding is perfectly lossless in HSL space.")
        elif hsl_vs_compressed_hsl_max_diff <= 1:
            st.info("✅ Near-perfect compression with minimal differences in HSL space.")
        elif hsl_vs_compressed_hsl_max_diff <= 5:
            st.warning("⚠️ Good compression with small differences in HSL space.")
        else:
            st.error("❌ Significant compression artifacts detected in HSL space.")
        
        # Show which conversion introduces the most error  
        if hsl_max_diff > 0 and hsl_vs_compressed_hsl_max_diff == 0:
            st.info("💡 **Analysis**: All precision loss comes from RGB→HSL→RGB conversion. Rectangular compression is perfect!")
        elif hsl_vs_compressed_hsl_max_diff > 0:
            st.warning("⚠️ **Analysis**: The rectangular compression introduces precision loss in HSL space.")
        else:
            st.success("✅ **Analysis**: Perfect preservation in both HSL space and RGB conversion.")
        
        # Show difference heatmaps for analysis
        if max(hsl_max_diff, compressed_max_diff, hsl_vs_compressed_hsl_max_diff) > 0:
            st.subheader("🔍 Difference Analysis")
            
            # Create difference heatmaps for all three comparisons
            def create_diff_heatmap(pixel_diffs, title):
                pixel_diff_squared = pixel_diffs.astype(np.float64)**2
                diff_sum = np.sum(pixel_diff_squared, axis=2)
                diff_sum = np.maximum(diff_sum, 0)
                diff_magnitude = np.sqrt(diff_sum)
                
                if np.max(diff_magnitude) > 0:
                    diff_heatmap = (diff_magnitude / np.max(diff_magnitude) * 255).astype(np.uint8)
                    heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)
                    heatmap_colored[:, :, 0] = diff_heatmap  # Red channel
                    return Image.fromarray(heatmap_colored), diff_magnitude
                return None, diff_magnitude
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if hsl_max_diff > 0:
                    heatmap_img, diff_mag = create_diff_heatmap(hsl_conversion_diff, "RGB vs HSL")
                    if heatmap_img:
                        st.image(heatmap_img, caption="RGB vs HSL Differences", use_column_width=True)
                else:
                    st.success("No differences in HSL conversion")
            
            with col2:
                if compressed_max_diff > 0:
                    heatmap_img, diff_mag = create_diff_heatmap(compressed_diff, "RGB vs Compressed")
                    if heatmap_img:
                        st.image(heatmap_img, caption="RGB vs Compressed Differences", use_column_width=True)
                else:
                    st.success("No differences in compression")
            
            with col3:
                if hsl_vs_compressed_hsl_max_diff > 0:
                    heatmap_img, diff_mag = create_diff_heatmap(hsl_vs_compressed_hsl_diff, "HSL vs Compressed HSL")
                    if heatmap_img:
                        st.image(heatmap_img, caption="HSL vs Compressed HSL", use_column_width=True)
                    
                    st.write(f"**Compression-Only Error Statistics:**")
                    st.write(f"- Maximum HSL difference: {np.max(diff_mag):.2f}")
                    st.write(f"- Mean HSL difference: {np.mean(diff_mag):.2f}")
                    st.write(f"- Pixels with differences: {np.sum(diff_mag > 0)} ({np.sum(diff_mag > 0)/total_pixels*100:.1f}%)")
                else:
                    st.success("Perfect rectangular compression!")